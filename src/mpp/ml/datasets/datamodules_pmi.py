# standard imports
import logging
from pathlib import Path

# third party imports
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# custom imports
from mpp.ml.datasets.tkms_pmi import TKMS_PMI_Dataset
from mpp.constants import PATHS

logging.basicConfig(
    format="%(asctime)s %(levelname)8s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)


class TKMS_PMI_DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for TKMS dataset with PMI features.
    
    This module handles the creation of train, validation, and test datasets
    with integrated PMI (Product Manufacturing Information) features.
    
    Parameters
    ----------
    batch_size : int, optional
        Batch size for dataloaders (default: 32).
    num_workers : int, optional
        Number of workers for dataloaders (default: 0).
    pin_memory : bool, optional
        Whether to pin memory for CUDA (default: True).
    pmi_path : str, optional
        Path to the encoded PMI features (.npy file).
    pmi_csv_path : str, optional
        Path to the PMI CSV file. If None, uses default location.
    clip_value : float or None, optional
        If provided, clips PMI features to [-clip_value, +clip_value].
    target_type : str, optional
        Type of prediction target ('step-set' or 'seq').
    input_type : str, optional
        Type of input features (currently only 'vecset').
    transform : callable, optional
        Optional transform for inputs.
    target_transform : callable, optional
        Optional transform for targets.
    """
    
    def __init__(
        self,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
        pmi_path="encoding_results/standard_encoding.npy",
        pmi_csv_path=None,  # Will use default if None
        clip_value=None,
        target_type="step-set",
        input_type="vecset",
        transform=None,
        target_transform=None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pmi_path = pmi_path
        self.pmi_csv_path = pmi_csv_path
        self.clip_value = clip_value
        self.target_type = target_type
        self.input_type = input_type
        self.transform = transform
        self.target_transform = target_transform
        
        # Set default CSV path if not provided
        if self.pmi_csv_path is None:
            # Try to find the CSV in the pmi_analyzer directory first
            pmi_analyzer_path = Path("/workspace/masterthesis_cadtoplan_fabian_heinze/pmi_analyzer/data/raw/manufacturing_features_with_processes.csv")
            if pmi_analyzer_path.exists():
                self.pmi_csv_path = str(pmi_analyzer_path)
                logger.info(f"Using PMI CSV from pmi_analyzer: {self.pmi_csv_path}")
            else:
                # Fallback to relative path
                self.pmi_csv_path = "pmi_analyzer/data/raw/manufacturing_features_with_processes.csv"
    
    def setup(self, stage=None):
        """
        Set up datasets for different stages.
        
        Parameters
        ----------
        stage : str, optional
            Stage name ('fit', 'validate', 'test', 'predict').
        """
        logger.info(f"Setting up TKMS PMI datamodule for stage: {stage}")
        
        if stage == "fit" or stage is None:
            self.train_dataset = TKMS_PMI_Dataset(
                mode="train",
                pmi_path=self.pmi_path,
                pmi_csv_path=self.pmi_csv_path,
                clip_value=self.clip_value,
                transform=self.transform,
                target_transform=self.target_transform,
                target_type=self.target_type,
                input_type=self.input_type
            )
            
            self.val_dataset = TKMS_PMI_Dataset(
                mode="valid",
                pmi_path=self.pmi_path,
                pmi_csv_path=self.pmi_csv_path,
                clip_value=self.clip_value,
                transform=self.transform,
                target_transform=self.target_transform,
                target_type=self.target_type,
                input_type=self.input_type
            )
            
            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            logger.info(f"Val dataset size: {len(self.val_dataset)}")
        
        if stage == "test" or stage is None:
            self.test_dataset = TKMS_PMI_Dataset(
                mode="test",
                pmi_path=self.pmi_path,
                pmi_csv_path=self.pmi_csv_path,
                clip_value=self.clip_value,
                transform=self.transform,
                target_transform=self.target_transform,
                target_type=self.target_type,
                input_type=self.input_type
            )
            
            logger.info(f"Test dataset size: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def predict_dataloader(self):
        """Return prediction dataloader (uses test dataset)."""
        return self.test_dataloader()


# Test the datamodule
if __name__ == "__main__":
    # Initialize datamodule
    dm = TKMS_PMI_DataModule(
        batch_size=16,
        target_type="step-set",
        pmi_path="encoding_results/standard_encoding.npy",
        pmi_csv_path="/workspace/masterthesis_cadtoplan_fabian_heinze/pmi_analyzer/data/raw/manufacturing_features_with_processes.csv",
        clip_value=5.0
    )
    
    # Setup datasets
    dm.setup(stage="fit")
    
    # Get dataloaders
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    # Test loading a batch
    for (vecset, pmi), target in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Vecset: {vecset.shape}")
        print(f"  PMI: {pmi.shape}")
        print(f"  Target: {target.shape}")
        
        # Check PMI statistics
        print(f"\nPMI batch stats:")
        print(f"  Min: {pmi.min().item():.3f}")
        print(f"  Max: {pmi.max().item():.3f}")
        print(f"  Mean: {pmi.mean().item():.3f}")
        print(f"  Non-zero: {(pmi != 0).sum().item()} / {pmi.numel()}")
        
        break  # Only check first batch