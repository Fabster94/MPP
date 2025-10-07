#!/usr/bin/env python3
"""
PMI Ablation Study for Manufacturing Process Classification

This script performs a systematic ablation study to determine the causal contribution
of different PMI feature groups to model performance.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from scipy.stats import wilcoxon

# Custom imports (adjust paths as needed)
from mpp.ml.models.classifier.cadtostepset_with_pmi import ProcessClassificationWithPMI
from mpp.ml.datasets.tkms import TKMS_Process_Dataset
from mpp.constants import PATHS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PMI Feature Group Definitions
# ============================================================================

def get_pmi_feature_order():
    """Define the exact order of PMI features as they appear in the encoded array"""
    # This MUST match your encoding order exactly!
    return [
        'total_dimension_count', 'linear_dimension_count', 
        'diameter_dimension_count', 'radius_dimension_count', 
        'angular_dimension_count', 'hole_fit_count', 'shaft_fit_count',
        'n_dia_tol', 'n_lin_tol',
        'tightest_dim_tol_lin_ultra_tight', 'tightest_dim_tol_lin_very_tight',
        'tightest_dim_tol_lin_tight', 'tightest_dim_tol_lin_medium',
        'tightest_dim_tol_lin_coarse', 'tightest_dim_tol_dia_ultra_tight',
        'tightest_dim_tol_dia_very_tight', 'tightest_dim_tol_dia_tight',
        'tightest_dim_tol_dia_medium', 'tightest_dim_tol_dia_coarse',
        'stats_dim_tol_lin_min', 'stats_dim_tol_lin_max', 'stats_dim_tol_lin_avg',
        'stats_dim_tol_dia_min', 'stats_dim_tol_dia_max', 'stats_dim_tol_dia_avg',
        'surface_spec_count', 'tightest_surface_very_fine',
        'tightest_surface_fine', 'tightest_surface_medium',
        'tightest_surface_rough', 'stats_surface_min',
        'stats_surface_max', 'stats_surface_avg',
        'has_angularity', 'has_circular_runout', 'has_concentricity',
        'has_cylindricity', 'has_flatness', 'has_parallelism',
        'has_perpendicularity', 'has_position', 'has_profile_of_line',
        'has_profile_of_surface', 'has_roundness', 'has_straightness',
        'has_symmetry', 'has_total_runout',
        'tightest_geom_tol_ultra_tight', 'tightest_geom_tol_very_tight',
        'tightest_geom_tol_tight', 'tightest_geom_tol_medium',
        'tightest_geom_tol_coarse', 'total_tolerance_count',
        'gtol_max_datum_refs', 'gtol_share_with_datum',
        'stats_geom_tol_min', 'stats_geom_tol_max', 'stats_geom_tol_avg'
    ]


def create_pmi_group_index_mapping():
    """Create mapping from group names to PMI feature indices"""
    feature_order = get_pmi_feature_order()
    feature_to_idx = {feat: idx for idx, feat in enumerate(feature_order)}
    
    # Define feature groups
    groups = {
        'dimensions': [
            'total_dimension_count', 'linear_dimension_count', 
            'diameter_dimension_count', 'radius_dimension_count', 
            'angular_dimension_count'
        ],
        'geometric_tolerances': [
            'has_angularity', 'has_circular_runout', 'has_concentricity',
            'has_cylindricity', 'has_flatness', 'has_parallelism',
            'has_perpendicularity', 'has_position', 'has_profile_of_line',
            'has_profile_of_surface', 'has_roundness', 'has_straightness',
            'has_symmetry', 'has_total_runout', 'tightest_geom_tol_ultra_tight',
            'tightest_geom_tol_very_tight', 'tightest_geom_tol_tight',
            'tightest_geom_tol_medium', 'tightest_geom_tol_coarse',
            'stats_geom_tol_min', 'stats_geom_tol_max', 'stats_geom_tol_avg'
        ],
        'dimensional_tolerances': [
            'n_dia_tol', 'n_lin_tol', 'tightest_dim_tol_lin_ultra_tight',
            'tightest_dim_tol_lin_very_tight', 'tightest_dim_tol_lin_tight',
            'tightest_dim_tol_lin_medium', 'tightest_dim_tol_lin_coarse',
            'tightest_dim_tol_dia_ultra_tight', 'tightest_dim_tol_dia_very_tight',
            'tightest_dim_tol_dia_tight', 'tightest_dim_tol_dia_medium',
            'tightest_dim_tol_dia_coarse', 'stats_dim_tol_lin_min',
            'stats_dim_tol_lin_max', 'stats_dim_tol_lin_avg',
            'stats_dim_tol_dia_min', 'stats_dim_tol_dia_max',
            'stats_dim_tol_dia_avg', 'total_tolerance_count'
        ],
        'surface_finish': [
            'surface_spec_count', 'tightest_surface_very_fine',
            'tightest_surface_fine', 'tightest_surface_medium',
            'tightest_surface_rough', 'stats_surface_min',
            'stats_surface_max', 'stats_surface_avg'
        ],
        'fits': ['hole_fit_count', 'shaft_fit_count'],
        'datums': ['gtol_max_datum_refs', 'gtol_share_with_datum']
    }
    
    # Convert to indices
    pmi_group_to_idxs = {}
    for group_name, features in groups.items():
        indices = []
        missing = []
        for f in features:
            if f in feature_to_idx:
                indices.append(feature_to_idx[f])
            else:
                missing.append(f)
        
        if missing:
            logger.warning(f"Group '{group_name}': Missing features {missing}")
        
        pmi_group_to_idxs[group_name] = indices
        logger.info(f"Group '{group_name}': {len(indices)} features, indices {indices[:3]}...")
    
    return pmi_group_to_idxs


# ============================================================================
# Extended TKMS Dataset with Masking Support
# ============================================================================

class TKMS_PMI_Dataset_Ablation(TKMS_Process_Dataset):
    """Extended TKMS dataset with PMI masking for ablation studies"""
    
    def __init__(self, mode="train", pmi_path=None, pmi_csv_path=None, 
                 clip_value=None, target_type="step-set",
                 mask_mode="none", mask_groups=None, mask_fill="mean", 
                 train_means=None, **kwargs):
        super().__init__(mode=mode, target_type=target_type, **kwargs)
        
        self.mask_mode = mask_mode  # "none", "minus", "only"
        self.mask_groups = mask_groups or []
        self.mask_fill = mask_fill  # "mean", "zero"
        self.train_means = train_means
        
        # Load PMI features
        self._load_pmi_data(pmi_path, pmi_csv_path, clip_value)
        
    def _load_pmi_data(self, pmi_path, pmi_csv_path, clip_value):
        """Load PMI features and create mapping"""
        # Load encoded PMI features
        pmi_full_path = Path(pmi_path)
        if not pmi_full_path.is_absolute():
            pmi_full_path = PATHS.ROOT / pmi_path
            
        self.pmi_features = torch.tensor(
            np.load(pmi_full_path), dtype=torch.float32
        )
        
        if clip_value is not None:
            self.pmi_features = torch.clamp(self.pmi_features, -clip_value, clip_value)
        
        # Load PMI dataframe for name mapping
        self.pmi_df = pd.read_csv(pmi_csv_path)
        
        # Determine key column
        key_col = 'part_name' if 'part_name' in self.pmi_df.columns else 'id'
        assert key_col in self.pmi_df.columns, f"'{key_col}' missing in PMI CSV"
        
        # CRITICAL: Build proper mapping from part_name to NPY row index
        # Assumption: NPY rows are in same order as CSV rows
        self.part_to_npyrow = {
            part_name: idx 
            for idx, part_name in enumerate(self.pmi_df[key_col].tolist())
        }
        
        # Create name mapping (handle _PMI suffix)
        self.name_mapping = {}
        for dataset_name in self.samples:
            csv_name = dataset_name[:-4] if dataset_name.endswith("_PMI") else dataset_name
            self.name_mapping[dataset_name] = csv_name
        
        logger.info(f"Loaded PMI features: shape {self.pmi_features.shape}")
        logger.info(f"Created mapping for {len(self.part_to_npyrow)} parts")
        
    def __getitem__(self, idx):
        """Get item with optional PMI masking"""
        vecset, target = super().__getitem__(idx)
        
        # Get PMI features
        dataset_name = self.samples[idx]
        csv_name = self.name_mapping.get(dataset_name, dataset_name)
        
        # Use proper mapping to get NPY row index
        npy_row_idx = self.part_to_npyrow.get(csv_name)
        
        if npy_row_idx is not None and npy_row_idx < len(self.pmi_features):
            pmi_vector = self.pmi_features[npy_row_idx].clone()
        else:
            if idx < 5:  # Only log first few
                logger.warning(f"No PMI data found for part: {dataset_name} -> {csv_name}")
            pmi_vector = torch.zeros(self.pmi_features.shape[1])
        
        # Apply masking
        if self.mask_mode != "none" and self.mask_groups:
            pmi_vector = self._apply_mask(pmi_vector)
            
        return (vecset, pmi_vector), target
    
    def _apply_mask(self, pmi_vector):
        """Apply group masking to PMI features"""
        pmi = pmi_vector.clone()
        
        if self.mask_mode == "minus":
            # Remove specific group
            if self.mask_fill == "zero":
                pmi[self.mask_groups] = 0.0
            elif self.mask_fill == "mean" and self.train_means is not None:
                pmi[self.mask_groups] = torch.tensor(
                    self.train_means[self.mask_groups], 
                    dtype=pmi.dtype
                )
                
        elif self.mask_mode == "only":
            # Keep only specific group
            mask = torch.ones(len(pmi), dtype=torch.bool)
            mask[self.mask_groups] = False
            
            if self.mask_fill == "zero":
                pmi[mask] = 0.0
            elif self.mask_fill == "mean" and self.train_means is not None:
                pmi[mask] = torch.tensor(
                    self.train_means[mask],
                    dtype=pmi.dtype
                )
                
        return pmi


# ============================================================================
# Training Functions
# ============================================================================

def compute_train_pmi_means(train_dataset, train_indices):
    """Compute mean PMI values across training set"""
    subset = Subset(train_dataset, train_indices)
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=4)
    
    pmi_sum = None
    count = 0
    
    for (_, pmi), _ in loader:
        if pmi_sum is None:
            pmi_sum = torch.zeros(pmi.shape[1])
        pmi_sum += pmi.sum(dim=0).cpu()
        count += pmi.shape[0]
    
    means = (pmi_sum / count).numpy()
    logger.info(f"Computed PMI means from {count} samples")
    return means


def find_best_threshold(model, loader, device='cuda'):
    """Find optimal F1 threshold on validation set"""
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for (vecset, pmi), y in loader:
            p = torch.sigmoid(model(vecset.to(device), pmi.to(device))).cpu().numpy()
            all_p.append(p)
            all_y.append(y.numpy())
    
    P = np.vstack(all_p)
    Y = np.vstack(all_y)
    
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        yhat = (P >= t).astype(int)
        f1 = f1_score(Y, yhat, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    
    logger.info(f"Best threshold: {best_t:.3f} (F1={best_f1:.4f})")
    return float(best_t)


def evaluate_model(model, loader, threshold=0.5, device='cuda', save_dir=None):
    """Evaluate model and return metrics, optionally save predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for (vecset, pmi), labels in loader:
            outputs = model(
                vecset.to(device), 
                pmi.to(device)
            )
            probs = torch.sigmoid(outputs)
            preds = (probs >= threshold).cpu()
            
            all_probs.append(probs.cpu())
            all_preds.append(preds)
            all_labels.append(labels)
    
    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Save predictions if directory specified
    if save_dir is not None:
        np.save(save_dir / "val_probs.npy", all_probs)
        np.save(save_dir / "val_labels.npy", all_labels)
        np.save(save_dir / "val_preds.npy", all_preds)
        logger.info(f"Saved validation predictions to {save_dir}")
    
    # Calculate metrics
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    
    # Per-class F1
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    return {
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'f1_bohren': float(f1_per_class[0]),
        'f1_drehen': float(f1_per_class[1]),
        'f1_fraesen': float(f1_per_class[2])
    }


def train_fold(fold: int, train_idx: np.ndarray, val_idx: np.ndarray, 
               args: argparse.Namespace, pmi_group_indices: Dict[str, List[int]],
               base_dir: Path) -> Dict:
    """Train one fold with ablation configuration"""
    
    # Set seeds for reproducibility  
    fold_seed = args.seed + fold
    seed_everything(fold_seed, workers=True)
    logger.info(f"Fold {fold}: Using seed {fold_seed} (base seed: {args.seed})")
    
    # Get mask indices if group specified
    mask_indices = []
    if args.group and args.ablation != 'none':
        if args.custom_indices:
            # Use custom indices if provided
            mask_indices = [int(idx) for idx in args.custom_indices.split(',')]
            logger.info(f"Fold {fold}: Using custom indices: {mask_indices}")
        elif args.group == 'ALL':
            # Special case for NO_PMI - mask all features
            pmi_dim_temp = TKMS_PMI_Dataset_Ablation(
                mode="train",
                pmi_path=args.pmi_path,
                pmi_csv_path=args.pmi_csv_path,
                clip_value=args.clip_value,
                target_type="step-set"
            ).pmi_features.shape[1]
            mask_indices = list(range(pmi_dim_temp))
            logger.info(f"Fold {fold}: NO_PMI mode - masking ALL {len(mask_indices)} PMI features")
        else:
            mask_indices = pmi_group_indices.get(args.group, [])
            logger.info(f"Fold {fold}: Ablation mode '{args.ablation}' for group '{args.group}' "
                       f"with {len(mask_indices)} features")
    
    # Create base dataset to compute means
    base_dataset = TKMS_PMI_Dataset_Ablation(
        mode="train",
        pmi_path=args.pmi_path,
        pmi_csv_path=args.pmi_csv_path,
        clip_value=args.clip_value,
        target_type="step-set"
    )
    
    # Compute train means (before masking!)
    train_means = compute_train_pmi_means(base_dataset, train_idx)
    
    # Create masked datasets
    train_dataset = TKMS_PMI_Dataset_Ablation(
        mode="train",
        pmi_path=args.pmi_path,
        pmi_csv_path=args.pmi_csv_path,
        clip_value=args.clip_value,
        target_type="step-set",
        mask_mode=args.ablation,
        mask_groups=mask_indices,
        mask_fill=args.mask_fill,
        train_means=train_means
    )
    
    # Use same dataset for validation but with different indices
    val_dataset = TKMS_PMI_Dataset_Ablation(
        mode="train",  # Same split file
        pmi_path=args.pmi_path,
        pmi_csv_path=args.pmi_csv_path,
        clip_value=args.clip_value,
        target_type="step-set",
        mask_mode=args.ablation,
        mask_groups=mask_indices,
        mask_fill=args.mask_fill,
        train_means=train_means
    )
    
    # Create data loaders
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=4, pin_memory=True)
    
    # Get PMI dimension from dataset
    pmi_dim = base_dataset.pmi_features.shape[1]
    logger.info(f"PMI dimension: {pmi_dim}")
    
    # Initialize model
    model = ProcessClassificationWithPMI(
        lr=args.lr,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        pmi_dim=int(pmi_dim)
    )
    
    # Setup training
    fold_dir = base_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=str(fold_dir),
        filename='best_model',
        save_top_k=1,
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min'
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_checkpointing=True,
        logger=False,  # Disable default logger
        enable_model_summary=False,
        deterministic=True,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Load best checkpoint
    best_model = ProcessClassificationWithPMI.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        pmi_dim=int(pmi_dim)
    )
    
    # Evaluate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_model = best_model.to(device)
    
    # Find optimal threshold on validation set
    threshold = find_best_threshold(best_model, val_loader, device=device)
    
    # Evaluate with optimal threshold and save predictions
    val_metrics = evaluate_model(best_model, val_loader, threshold=threshold, 
                                 device=device, save_dir=fold_dir)
    
    # Save fold configuration
    config = {
        'fold': fold,
        'ablation_mode': args.ablation,
        'ablation_group': args.group,
        'mask_indices': mask_indices,
        'mask_fill': args.mask_fill,
        'train_means': train_means.tolist(),
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'best_epoch': trainer.current_epoch,
        'val_threshold': threshold,
        'metrics': val_metrics,
        'pmi_dim': int(pmi_dim),
        'pmi_groups': pmi_group_indices,
        'best_model_path': checkpoint_callback.best_model_path,
        'hyperparameters': {
            'lr': args.lr,
            'embed_dim': args.embed_dim,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'dropout': args.dropout,
            'weight_decay': args.weight_decay,
            'max_epochs': args.max_epochs,
            'batch_size': args.batch_size
        }
    }
    
    with open(fold_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Fold {fold} completed: F1-macro={val_metrics['f1_macro']:.4f}")
    
    return val_metrics


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PMI Ablation Study')
    
    # Ablation settings
    parser.add_argument('--ablation', choices=['none', 'minus', 'only'], 
                       default='none', help='Ablation mode')
    parser.add_argument('--group', type=str, help='PMI group name to ablate')
    parser.add_argument('--custom_indices', type=str, 
                       help='Comma-separated list of feature indices for custom ablation')
    parser.add_argument('--mask_fill', choices=['mean', 'zero'], 
                       default='mean', help='How to fill masked features')
    parser.add_argument('--tag', type=str, required=True, 
                       help='Experiment tag (e.g., FULL, MINUS_dimensions)')
    
    # Data settings
    parser.add_argument('--pmi_path', type=str, 
                       default='/workspace/masterthesis_cadtoplan_fabian_heinze/mpp/encoding_results/standard_encoding.npy')
    parser.add_argument('--pmi_csv_path', type=str,
                       default='/workspace/masterthesis_cadtoplan_fabian_heinze/pmi_analyzer/data/raw/manufacturing_features_with_processes.csv')
    parser.add_argument('--clip_value', type=float, default=5.0)
    
    # Model settings (use best from tuning)
    parser.add_argument('--lr', type=float, default=0.00178)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.338)
    parser.add_argument('--weight_decay', type=float, default=1.86729e-05)
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=85)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--n_folds', type=int, default=5)
    
    # Output settings
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/ablation'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Base random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    # IMPORTANT: Don't create timestamp subdirectory!
    # Use the output_dir directly as passed from cli.py
    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Save args
    args_dict = vars(args)
    args_dict['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    args_dict['base_seed'] = args.seed
    with open(base_dir / 'args.json', 'w') as f:
        json.dump(args_dict, f, indent=2)
    
    logger.info(f"Starting ablation experiment: {args.tag}")
    logger.info(f"Output directory: {base_dir}")
    
    # Get PMI group mappings
    pmi_group_indices = create_pmi_group_index_mapping()
    
    # Load dataset for splitting
    dataset = TKMS_PMI_Dataset_Ablation(
        mode="train",
        pmi_path=args.pmi_path,
        pmi_csv_path=args.pmi_csv_path,
        clip_value=args.clip_value,
        target_type="step-set"
    )
    
    # Try to use MultilabelStratifiedKFold for better label distribution
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        
        # Get all labels as numpy array
        all_labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            all_labels.append(label.numpy())
        
        Y = np.stack(all_labels)
        skf = MultilabelStratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        logger.info("Using MultilabelStratifiedKFold for better label distribution")
        
    except ImportError:
        logger.warning("iterative-stratification not installed, falling back to StratifiedKFold")
        
        # Get all labels for stratification
        all_labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            # For multilabel, use the first positive class for stratification
            first_class = label.nonzero()[0][0].item() if label.sum() > 0 else 0
            all_labels.append(first_class)
        
        Y = all_labels
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(Y, Y)):
        logger.info(f"\nStarting fold {fold+1}/{args.n_folds}")
        
        metrics = train_fold(fold, train_idx, val_idx, args, pmi_group_indices, base_dir)
        fold_metrics.append(metrics)
    
    # Aggregate results
    results = {
        'experiment': args.tag,
        'ablation_mode': args.ablation,
        'ablation_group': args.group,
        'mask_fill': args.mask_fill,
        'fold_metrics': fold_metrics,
        'aggregated': {
            'f1_macro': {
                'mean': np.mean([m['f1_macro'] for m in fold_metrics]),
                'std': np.std([m['f1_macro'] for m in fold_metrics]),
                'values': [m['f1_macro'] for m in fold_metrics]
            },
            'f1_micro': {
                'mean': np.mean([m['f1_micro'] for m in fold_metrics]),
                'std': np.std([m['f1_micro'] for m in fold_metrics]),
                'values': [m['f1_micro'] for m in fold_metrics]
            }
        }
    }
    
    # Add per-class aggregation
    for class_name in ['bohren', 'drehen', 'fraesen']:
        key = f'f1_{class_name}'
        results['aggregated'][key] = {
            'mean': np.mean([m[key] for m in fold_metrics]),
            'std': np.std([m[key] for m in fold_metrics]),
            'values': [m[key] for m in fold_metrics]
        }
    
    # Save final results
    with open(base_dir / 'final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nExperiment completed: {args.tag}")
    logger.info(f"F1-macro: {results['aggregated']['f1_macro']['mean']:.4f} "
               f"± {results['aggregated']['f1_macro']['std']:.4f}")


if __name__ == "__main__":
    main()