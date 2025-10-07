# standard imports
import logging
from pathlib import Path

# third party imports
import torch
import numpy as np
import pandas as pd

# custom imports
from mpp.constants import PATHS, TKMS_VOCAB, INV_TKMS_VOCAB
from mpp.ml.datasets.tkms import TKMS_Process_Dataset

logger = logging.getLogger(__name__)


class TKMS_PMI_Dataset(TKMS_Process_Dataset):
    """
    TKMS Dataset with integrated PMI (Product Manufacturing Information) features.
    
    This dataset extends the base TKMS_Process_Dataset by adding pre-encoded PMI features
    to each sample. The PMI features provide additional manufacturing constraints like
    tolerances, surface specifications, and geometric requirements.
    
    Parameters
    ----------
    mode : str, optional
        Dataset split to use. One of {'train', 'valid', 'test'}. Default is 'train'.
    pmi_path : str, optional
        Path to the encoded PMI features (.npy file). Can be absolute or relative to project root.
        Default is 'encoding_results/standard_encoding.npy'.
    pmi_csv_path : str, optional
        Path to the PMI CSV file with part names. If None, tries to find it automatically.
    clip_value : float or None, optional
        If provided, clips PMI features to [-clip_value, +clip_value]. Default is None (no clipping).
    transform : callable, optional
        Optional transform to be applied on the input sample.
    target_transform : callable, optional
        Optional transform to be applied on the target.
    target_type : str, optional
        Type of prediction target. One of {'step-set', 'seq'}.
    input_type : str, optional
        Type of input features. Currently only 'vecset' is supported.
    
    Raises
    ------
    FileNotFoundError
        If the PMI features file cannot be found at the specified path.
    ValueError
        If the number of PMI samples doesn't match the dataset size.
    """
    
    def __init__(
        self,
        mode="train",
        pmi_path="encoding_results/standard_encoding.npy",
        pmi_csv_path=None,
        clip_value=None,
        transform=None,
        target_transform=None,
        target_type="step-set",
        input_type="vecset"
    ):
        # Initialize parent dataset
        super().__init__(mode, transform, target_transform, target_type, input_type)
        
        # Load PMI DataFrame first (for name mapping)
        self._load_pmi_dataframe(pmi_csv_path)
        
        # Create name mapping
        self._create_name_mapping()
        
        # Load PMI features
        self._load_pmi_features(pmi_path, clip_value)
        
        # Validate data consistency
        self._validate_pmi_data()
        
    def _load_pmi_dataframe(self, pmi_csv_path=None):
        """Load the PMI dataframe that contains part_name mappings"""
        # Try different paths
        possible_paths = []
        
        if pmi_csv_path:
            possible_paths.append(Path(pmi_csv_path))
        
        # Add default locations
        possible_paths.extend([
            Path("/workspace/masterthesis_cadtoplan_fabian_heinze/mpp/data_pmi/pmi_features.csv")
        ])
        
        # Find the first existing path
        csv_path = None
        for path in possible_paths:
            if path.exists():
                csv_path = path
                break
        
        if not csv_path:
            raise FileNotFoundError(
                f"PMI DataFrame not found. Tried paths:\n" + 
                "\n".join(str(p) for p in possible_paths)
            )
        
        logger.info(f"Loading PMI DataFrame from: {csv_path}")
        self.pmi_df = pd.read_csv(csv_path)
        
        # Verify required columns exist
        if 'part_name' not in self.pmi_df.columns:
            raise ValueError("PMI DataFrame must contain 'part_name' column")
        
        logger.info(f"Loaded PMI data for {len(self.pmi_df)} parts")
    
    def _create_name_mapping(self):
        """Create mapping between dataset names and CSV names"""
        self.name_mapping = {}
        
        # Show sample names for debugging
        logger.info(f"Sample dataset names: {self.samples[:3] if len(self.samples) > 3 else self.samples}")
        logger.info(f"Sample CSV names: {list(self.pmi_df['part_name'].values[:3])}")
        
        # Strategy 1: Direct mapping (dataset names match CSV names)
        for dataset_name in self.samples:
            self.name_mapping[dataset_name] = dataset_name
        
        # Check match rate
        matched = sum(1 for csv_name in self.name_mapping.values() 
                     if csv_name in self.pmi_df['part_name'].values)
        
        if matched == 0:
            # Strategy 2: Remove _PMI suffix if present
            if self.samples[0].endswith("_PMI"):
                logger.info("No direct matches found. Removing '_PMI' suffix from dataset names...")
                self.name_mapping = {}
                for dataset_name in self.samples:
                    csv_name = dataset_name[:-4] if dataset_name.endswith("_PMI") else dataset_name
                    self.name_mapping[dataset_name] = csv_name
                
                matched = sum(1 for csv_name in self.name_mapping.values() 
                             if csv_name in self.pmi_df['part_name'].values)
        
        logger.info(f"Name mapping created: {matched}/{len(self.samples)} parts found in PMI data")
        
        if matched == 0:
            logger.error("No matching parts found! Please check name formats:")
            logger.error(f"Dataset examples: {list(self.samples)[:5]}")
            logger.error(f"CSV examples: {list(self.pmi_df['part_name'].values)[:5]}")
            raise ValueError("No matching parts found between dataset and PMI data")
        
    def _load_pmi_features(self, pmi_path, clip_value):
        """Load and optionally clip PMI features"""
        # Resolve path
        pmi_full_path = Path(pmi_path)
        if not pmi_full_path.is_absolute():
            pmi_full_path = PATHS.ROOT / pmi_path
        
        if not pmi_full_path.exists():
            raise FileNotFoundError(
                f"PMI features file not found at: {pmi_full_path}\n"
                f"Please run PMI encoding first or provide correct path."
            )
        
        # Load features
        logger.info(f"Loading PMI features from: {pmi_full_path}")
        pmi_array = np.load(pmi_full_path)
        
        # Convert to tensor
        self.pmi_features = torch.tensor(pmi_array, dtype=torch.float32)
        
        # Apply clipping if requested
        if clip_value is not None:
            logger.info(f"Clipping PMI features to ±{clip_value}")
            self.pmi_features = torch.clamp(self.pmi_features, -clip_value, clip_value)
        
        logger.info(f"Loaded PMI features with shape: {self.pmi_features.shape}")
        
        # Store metadata
        self.pmi_dim = self.pmi_features.shape[1]
        self.clip_value = clip_value
        
    def _validate_pmi_data(self):
        """Check how many parts have PMI data"""
        matched_parts = 0
        missing_parts = []
        
        for dataset_name in self.samples:
            csv_name = self.name_mapping.get(dataset_name, dataset_name)
            if csv_name in self.pmi_df['part_name'].values:
                matched_parts += 1
            else:
                missing_parts.append(dataset_name)
        
        logger.info(
            f"PMI data coverage: {matched_parts}/{len(self.samples)} parts "
            f"({matched_parts/len(self.samples)*100:.1f}%)"
        )
        
        if missing_parts and len(missing_parts) <= 10:
            logger.warning(f"Missing PMI data for: {missing_parts}")
        elif missing_parts:
            logger.warning(f"Missing PMI data for {len(missing_parts)} parts")
    
    def __getitem__(self, idx):
        """
        Returns the indexed sample with both vecset and PMI features.
        
        Parameters
        ----------
        idx : int
            Index of the sample.
        
        Returns
        -------
        tuple((torch.Tensor, torch.Tensor), torch.Tensor)
            ((vecset, pmi_features), target) where:
            - vecset: Geometry features of shape (set_size, input_dim)
            - pmi_features: PMI features of shape (pmi_dim,)
            - target: Process labels (format depends on target_type)
        """
        # Get vecset and target from parent class
        vecset, target = super().__getitem__(idx)
        
        # Get part name for this index
        dataset_name = self.samples[idx]
        csv_name = self.name_mapping.get(dataset_name, dataset_name)
        
        # Find PMI data by part name
        pmi_rows = self.pmi_df[self.pmi_df['part_name'] == csv_name]
        
        if len(pmi_rows) > 0:
            # Get the row index in the preprocessed features
            pmi_row_idx = pmi_rows.index[0]
            
            # Ensure index is within bounds
            if pmi_row_idx < len(self.pmi_features):
                pmi_vector = self.pmi_features[pmi_row_idx]
            else:
                logger.warning(f"PMI index {pmi_row_idx} out of bounds for part: {dataset_name}")
                pmi_vector = torch.zeros(self.pmi_dim)
        else:
            # Part not found in PMI data - use zero vector
            if idx < 5:  # Only log first few to avoid spam
                logger.warning(f"No PMI data found for part: {dataset_name} (CSV name: {csv_name})")
            pmi_vector = torch.zeros(self.pmi_dim)
        
        return (vecset, pmi_vector), target
    
    def get_pmi_statistics(self):
        """
        Get statistics about the PMI features in this dataset.
        
        Returns
        -------
        dict
            Dictionary containing min, max, mean, std of PMI features
        """
        return {
            'min': self.pmi_features.min().item(),
            'max': self.pmi_features.max().item(),
            'mean': self.pmi_features.mean().item(),
            'std': self.pmi_features.std().item(),
            'shape': tuple(self.pmi_features.shape),
            'clipped': self.clip_value is not None,
            'clip_value': self.clip_value
        }
    
    def get_sample_info(self, idx):
        """
        Get detailed information about a specific sample.
        
        Parameters
        ----------
        idx : int
            Sample index
            
        Returns
        -------
        dict
            Dictionary with part_name, processes, vecset shape, and PMI stats
        """
        (vecset, pmi), target = self[idx]
        
        dataset_name = self.samples[idx]
        csv_name = self.name_mapping.get(dataset_name, dataset_name)
        
        return {
            'dataset_name': dataset_name,
            'csv_name': csv_name,
            'processes': self.decode_sequence(target.tolist()) if self.target_type == 'seq' else 'multi-label',
            'vecset_shape': tuple(vecset.shape),
            'pmi_shape': tuple(pmi.shape),
            'pmi_min': pmi.min().item(),
            'pmi_max': pmi.max().item(),
            'pmi_mean': pmi.mean().item(),
            'has_pmi': pmi.sum().item() != 0  # Check if PMI is not all zeros
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the dataset
    for mode in ['train', 'valid', 'test']:
        try:
            dataset = TKMS_PMI_Dataset(
                mode=mode,
                target_type="step-set",
                pmi_path="/workspace/masterthesis_cadtoplan_fabian_heinze/mpp/encoding_results/standard_encoding.npy",
                pmi_csv_path="/workspace/masterthesis_cadtoplan_fabian_heinze/pmi_analyzer/data/raw/manufacturing_features_with_processes.csv"
            )
            
            # Get a sample
            (vecset, pmi), target = dataset[0]
            
            logger.info(f"\n{mode.upper()} Dataset:")
            logger.info(f"  Dataset size: {len(dataset)}")
            logger.info(f"  Vecset shape: {vecset.shape}")
            logger.info(f"  PMI shape: {pmi.shape}")
            logger.info(f"  Target shape: {target.shape}")
            
            # Show PMI statistics
            stats = dataset.get_pmi_statistics()
            logger.info(f"  PMI stats: min={stats['min']:.3f}, max={stats['max']:.3f}, "
                       f"mean={stats['mean']:.3f}, std={stats['std']:.3f}")
            
            # Check a few samples
            logger.info("\n  Sample checks:")
            for i in range(min(5, len(dataset))):
                info = dataset.get_sample_info(i)
                logger.info(f"    {i}: {info['dataset_name']} -> {info['csv_name']}, has_pmi: {info['has_pmi']}")
            
        except FileNotFoundError as e:
            logger.error(f"Error loading {mode} dataset: {e}")
            logger.info("Please ensure PMI features are encoded first using the encoding script.")
    
    # Test with clipping
    logger.info("\nTesting with clipping:")
    dataset_clipped = TKMS_PMI_Dataset(
        mode="train",
        target_type="step-set", 
        clip_value=5.0
    )
    stats_clipped = dataset_clipped.get_pmi_statistics()
    logger.info(f"  Clipped PMI stats: min={stats_clipped['min']:.3f}, max={stats_clipped['max']:.3f}")