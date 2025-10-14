#!/usr/bin/env python3
"""
PMI Feature Ablation Study with Repeated Cross-Validation
Uses mean-masking and focuses on WITHOUT_X ablations
Includes FULL and GEOMETRY_ONLY controls for proper paired comparisons
"""

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import f1_score, jaccard_score, accuracy_score, precision_recall_curve
from torch.utils.data import DataLoader, Subset, ConcatDataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import logging
import warnings
warnings.filterwarnings('ignore')

# PyTorch Lightning
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Setup logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Multilabel stratification - REQUIRED
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

# Custom imports
from mpp.ml.models.classifier.unified_process_classifier import UnifiedProcessClassifier
from mpp.ml.datasets.tkms import TKMS_Process_Dataset
from mpp.ml.datasets.tkms_pmi import TKMS_PMI_Dataset

# ========== CONFIGURATION ==========
N_FOLDS = 5
N_REPEATS = 5  # Match main CV
SEED = 42
BATCH_SIZE = 85
MAX_EPOCHS = 100
PATIENCE = 20
CLASS_NAMES = ["Bohren", "Drehen", "Fräsen"]
NUM_WORKERS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Statistical parameters
N_BOOTSTRAP = 5000
ALPHA = 0.05

# Output Directory
OUTPUT_DIR = Path("ablation_results_cv") / datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Re-configure logging with file handler after output dir exists
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(OUTPUT_DIR / 'ablation_log.txt')),
        logging.StreamHandler()
    ],
    force=True  # Override previous config
)
logger = logging.getLogger(__name__)

# Best hyperparameters from main CV
HP_GEOM = {
    "dropout": 0.224,
    "lr": 0.000326,
    "embed_dim": 128,
    "num_layers": 2,
    "num_heads": 16,
    "weight_decay": 0.000374,
    "use_pmi": False,  # Geometry-only
    "pmi_dim": 30,  # Still needed for model initialization
    "initial_gate": 0.2,
    "modality_dropout": 0.0,
    "max_epochs": MAX_EPOCHS
}

HP_PMI = {
    "dropout": 0.280,
    "lr": 0.000690,
    "embed_dim": 64,
    "num_layers": 3,
    "num_heads": 8,
    "weight_decay": 0.000277,
    "use_pmi": True,
    "pmi_dim": 30,
    "initial_gate": 0.171,
    "modality_dropout": 0.206,
    "max_epochs": MAX_EPOCHS
}

# PMI Configuration
PMI_CONFIG = {
    "pmi_path": "/workspace/masterthesis_cadtoplan_fabian_heinze/mpp/encoding_results/standard_encoding.npy",
    "pmi_csv_path": "/workspace/masterthesis_cadtoplan_fabian_heinze/pmi_analyzer/data/raw/manufacturing_features_with_processes.csv",
    "clip_value": 5.0
}


def ensure_determinism(seed):
    """Ensure reproducible results"""
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        logger.warning("Could not enable fully deterministic algorithms.")


def get_pmi_feature_groups():
    """Define PMI feature groups with correct indices"""
    # Feature groups based on your categorization
    feature_groups = {
        'dimensions': [0, 1, 2, 3, 4],  # 5 features
        'dimensional_tolerances': [5, 6, 7, 8, 9, 10, 11, 12, 13],  # 9 features
        'geometric_tolerances': [14, 15, 16, 17, 18, 19, 20, 21],  # 8 features
        'surface_finish': [22, 23, 24, 25],  # 4 features
        'fits': [26, 27],  # 2 features
        'datums': [28, 29]  # 2 features
    }
    
    # Verify all 30 features are covered
    all_indices = []
    for indices in feature_groups.values():
        all_indices.extend(indices)
    assert len(set(all_indices)) == 30, f"Feature mapping error: {len(set(all_indices))} != 30"
    
    logger.info("Feature groups defined:")
    for group, indices in feature_groups.items():
        logger.info(f"  {group}: {len(indices)} features (indices {indices[0]}-{indices[-1]})")
    
    return feature_groups


class PMI_Ablation_Dataset(TKMS_PMI_Dataset):
    """Extended dataset with mean-masking for ablation"""
    
    def __init__(self, mask_indices=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_indices = mask_indices if mask_indices is not None else []
        
        # Compute mean values for masking
        if len(self.mask_indices) > 0:
            self.mean_values = self.pmi_features.mean(dim=0)
            logger.debug(f"Computed mean values for masking {len(self.mask_indices)} features")
    
    def __getitem__(self, idx):
        # Get the original item from parent class
        (vecset, pmi_original), label = super().__getitem__(idx)
        
        # Clone PMI features for masking
        pmi_tensor = pmi_original.clone().float()
        
        # Apply mean-masking
        if len(self.mask_indices) > 0:
            pmi_tensor[self.mask_indices] = self.mean_values[self.mask_indices]
        
        return (vecset, pmi_tensor), label


def find_optimal_thresholds_on_train(model, train_subset, dataset, labels_augmented, train_indices, use_pmi=True, per_class=True, val_split=0.1, fold_seed=42):
    """Find optimal thresholds on inner train/val split to avoid leakage - matching CV script"""
    # Get augmented labels for stratified split
    y_train_augmented = labels_augmented[train_indices]
    
    # Create stratified inner split - always use MultilabelStratifiedShuffleSplit
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=fold_seed)
    inner_train_rel, inner_val_rel = next(msss.split(np.zeros(len(y_train_augmented)), y_train_augmented))
    
    # Map back to original indices
    inner_val_indices = [train_indices[i] for i in inner_val_rel]
    
    # Create inner validation loader
    inner_val_subset = Subset(dataset, inner_val_indices)
    inner_val_loader = DataLoader(
        inner_val_subset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Find thresholds on inner validation set
    all_probs = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in inner_val_loader:
            if use_pmi:
                (inputs, pmi), labels = batch
                outputs = model(inputs.to(DEVICE), pmi.to(DEVICE))
            else:
                inputs, labels = batch
                outputs = model(inputs.to(DEVICE))
            
            probs = torch.sigmoid(outputs).cpu()
            all_probs.append(probs)
            all_labels.append(labels)
    
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    if per_class:
        # Find per-class thresholds
        thresholds = np.zeros(len(CLASS_NAMES))
        for class_idx in range(len(CLASS_NAMES)):
            precision, recall, thresholds_pr = precision_recall_curve(
                all_labels[:, class_idx], 
                all_probs[:, class_idx]
            )
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            if best_idx < len(thresholds_pr):
                thresholds[class_idx] = thresholds_pr[best_idx]
            else:
                thresholds[class_idx] = 0.5
    else:
        # Global threshold
        best_threshold = 0.5
        best_f1 = 0.0
        for threshold in np.arange(0.1, 0.9, 0.01):
            preds = (all_probs > threshold).astype(int)
            f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds = best_threshold
    
    logger.debug(f"Optimal thresholds found: {thresholds}")
    return thresholds


def calculate_bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, seed=42):
    """Calculate bootstrap confidence interval for paired differences"""
    rng = np.random.default_rng(seed)
    boot_means = []
    
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(values), len(values))
        boot_means.append(values[idx].mean())
    
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return values.mean(), ci_low, ci_high


def train_ablation_fold(repeat_idx, fold_idx, train_idx, val_idx, dataset, labels_augmented, ablation_name, hyperparameters):
    """Train one fold of ablation variant - matching CV script approach"""
    global_fold_idx = repeat_idx * N_FOLDS + fold_idx
    fold_seed = SEED + global_fold_idx
    ensure_determinism(fold_seed)
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Initialize model with specified hyperparameters
    model = UnifiedProcessClassifier(**hyperparameters)
    
    # Setup training
    checkpoint_dir = OUTPUT_DIR / f"repeat_{repeat_idx}" / f"fold_{fold_idx}" / ablation_name
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best',
        monitor='val_loss',
        save_top_k=1,
        mode='min'
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        mode='min'
    )
    
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stop],
        enable_progress_bar=True,  # Show training progress
        logger=False,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        gradient_clip_val=1.0,
        deterministic=True
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model
    model = UnifiedProcessClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)
    model.to(DEVICE)
    model.eval()
    
    # Find optimal thresholds matching CV script approach
    use_pmi = hyperparameters.get('use_pmi', True)
    thresholds = find_optimal_thresholds_on_train(
        model, train_subset, dataset, labels_augmented, train_idx,
        use_pmi=use_pmi, per_class=True, fold_seed=fold_seed
    )
    
    # Evaluate with optimal thresholds
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            if use_pmi:
                # PMI model expects tuple of (vecset, pmi)
                (vecset, pmi), labels = batch
                outputs = model(vecset.to(DEVICE), pmi.to(DEVICE))
            else:
                # Geometry-only model expects just vecset
                vecset, labels = batch
                outputs = model(vecset.to(DEVICE))
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Apply per-class thresholds
            if isinstance(thresholds, np.ndarray):
                preds = np.zeros_like(probs, dtype=int)
                for i in range(len(CLASS_NAMES)):
                    preds[:, i] = (probs[:, i] > thresholds[i]).astype(int)
            else:
                preds = (probs > thresholds).astype(int)
            
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics
    metrics = {
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'jaccard_samples': jaccard_score(all_labels, all_preds, average='samples', zero_division=0),
        'subset_accuracy': accuracy_score(all_labels, all_preds),
        'repeat': repeat_idx,
        'fold': fold_idx,
        'epochs_trained': trainer.current_epoch + 1,
        'thresholds': thresholds.tolist() if isinstance(thresholds, np.ndarray) else [thresholds] * 3
    }
    
    # Per-class metrics
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    for i, class_name in enumerate(CLASS_NAMES):
        metrics[f'f1_{class_name.lower()}'] = f1_per_class[i]
    
    # Track gate value if available and PMI is used
    if use_pmi and hasattr(model, 'gate'):
        metrics['gate_value'] = torch.sigmoid(model.gate).item()
    
    return metrics


def load_aligned_datasets():
    """Load and align datasets ensuring same order - matching CV script approach"""
    logger.info("Loading and aligning datasets...")
    
    # Load geometry datasets (for GEOMETRY_ONLY)
    train_geom = TKMS_Process_Dataset(mode="train", target_type="step-set")
    valid_geom = TKMS_Process_Dataset(mode="valid", target_type="step-set")
    
    # Load PMI datasets (for FULL and WITHOUT_X)
    train_pmi = TKMS_PMI_Dataset(mode="train", target_type="step-set", **PMI_CONFIG)
    valid_pmi = TKMS_PMI_Dataset(mode="valid", target_type="step-set", **PMI_CONFIG)
    
    # Verify sample order alignment
    assert train_pmi.samples == train_geom.samples, "Train sample order mismatch!"
    assert valid_pmi.samples == valid_geom.samples, "Valid sample order mismatch!"
    
    all_sample_ids = train_pmi.samples + valid_pmi.samples
    
    # Create combined datasets
    dataset_geom = ConcatDataset([train_geom, valid_geom])
    dataset_pmi = ConcatDataset([train_pmi, valid_pmi])
    
    # Get labels for stratification (from either dataset, they're aligned)
    all_labels = []
    for i in range(len(train_geom) + len(valid_geom)):
        if i < len(train_geom):
            _, label = train_geom[i]
        else:
            _, label = valid_geom[i - len(train_geom)]
        all_labels.append(label.numpy())
    labels = np.array(all_labels)
    
    # Add interaction columns for better stratification
    labels_int = labels.astype(int)
    interactions = np.zeros((len(labels), 3), dtype=int)
    interactions[:, 0] = labels_int[:, 0] & labels_int[:, 1]  # B∧D
    interactions[:, 1] = labels_int[:, 0] & labels_int[:, 2]  # B∧F
    interactions[:, 2] = labels_int[:, 1] & labels_int[:, 2]  # D∧F
    labels_augmented = np.hstack([labels_int, interactions])
    
    logger.info(f"Total samples: {len(labels)}")
    logger.info(f"Label distribution: {labels.sum(axis=0)} ({CLASS_NAMES})")
    
    return dataset_geom, dataset_pmi, labels, labels_augmented, all_sample_ids


def generate_all_splits(labels_augmented, n_repeats=N_REPEATS, n_folds=N_FOLDS, seed=SEED):
    """Pre-generate all CV splits to ensure perfect pairing across ablations"""
    all_splits = {}
    
    for repeat_idx in range(n_repeats):
        repeat_seed = seed + repeat_idx * 1000
        
        # Always use stratified splits - iterstrat is required
        splitter = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=repeat_seed)
        splits = list(splitter.split(np.zeros(len(labels_augmented)), labels_augmented))
        
        all_splits[repeat_idx] = splits
        logger.debug(f"Generated splits for repeat {repeat_idx}: {n_folds} folds")
    
    return all_splits


def run_without_x_ablations():
    """Run WITHOUT_X ablations with repeated CV including FULL and GEOMETRY_ONLY controls"""
    logger.info("="*80)
    logger.info("PMI FEATURE ABLATION STUDY - WITHOUT_X Analysis")
    logger.info(f"Configuration: {N_REPEATS}×{N_FOLDS} = {N_REPEATS*N_FOLDS} total folds")
    logger.info("="*80)
    
    # Get feature groups
    feature_groups = get_pmi_feature_groups()
    
    # Load aligned datasets (both geometry and PMI)
    dataset_geom, dataset_pmi, labels, labels_augmented, sample_ids = load_aligned_datasets()
    
    # Pre-generate all splits for perfect pairing
    logger.info("\nPre-generating CV splits for paired comparisons...")
    all_splits = generate_all_splits(labels_augmented)
    
    # Save splits for reproducibility
    splits_for_save = {}
    for repeat_idx, splits in all_splits.items():
        splits_for_save[f"repeat_{repeat_idx}"] = [
            {"train": train_idx.tolist(), "val": val_idx.tolist()} 
            for train_idx, val_idx in splits
        ]
    with open(OUTPUT_DIR / 'cv_splits.json', 'w') as f:
        json.dump(splits_for_save, f, indent=2)
    logger.info(f"Saved CV splits to {OUTPUT_DIR / 'cv_splits.json'}")
    
    # Results storage
    all_results = {}
    
    # 1. Run GEOMETRY_ONLY baseline using TKMS_Process_Dataset
    logger.info("\n" + "="*60)
    logger.info("Running: GEOMETRY_ONLY (Baseline - No PMI)")
    logger.info("Using TKMS_Process_Dataset with HP_GEOM hyperparameters")
    logger.info("="*60)
    
    geom_results = []
    
    for repeat_idx in range(N_REPEATS):
        for fold_idx, (train_idx, val_idx) in enumerate(all_splits[repeat_idx]):
            logger.info(f"  GEOMETRY_ONLY - Repeat {repeat_idx+1}/{N_REPEATS}, Fold {fold_idx+1}/{N_FOLDS}")
            
            metrics = train_ablation_fold(
                repeat_idx, fold_idx, train_idx, val_idx, 
                dataset_geom, labels_augmented, 'GEOMETRY_ONLY', HP_GEOM
            )
            geom_results.append(metrics)
            logger.info(f"    F1-Macro: {metrics['f1_macro']:.4f}, Epochs: {metrics['epochs_trained']}")
    
    all_results['GEOMETRY_ONLY'] = geom_results
    
    # 2. Run FULL control condition using PMI_Ablation_Dataset (no masking)
    logger.info("\n" + "="*60)
    logger.info("Running: FULL (Control - All PMI Features)")
    logger.info("Using PMI_Ablation_Dataset with HP_PMI hyperparameters")
    logger.info("="*60)
    
    full_results = []
    
    # Create unmasked PMI dataset for FULL condition
    full_dataset = PMI_Ablation_Dataset(
        mode="train", target_type="step-set",
        mask_indices=[],  # No masking
        **PMI_CONFIG
    )
    valid_dataset = PMI_Ablation_Dataset(
        mode="valid", target_type="step-set",
        mask_indices=[],
        **PMI_CONFIG
    )
    full_combined = ConcatDataset([full_dataset, valid_dataset])
    
    # Run FULL with pre-generated splits
    for repeat_idx in range(N_REPEATS):
        for fold_idx, (train_idx, val_idx) in enumerate(all_splits[repeat_idx]):
            logger.info(f"  FULL - Repeat {repeat_idx+1}/{N_REPEATS}, Fold {fold_idx+1}/{N_FOLDS}")
            
            metrics = train_ablation_fold(
                repeat_idx, fold_idx, train_idx, val_idx, 
                full_combined, labels_augmented, 'FULL', HP_PMI
            )
            full_results.append(metrics)
            gate_str = f", Gate: {metrics.get('gate_value', 'N/A'):.3f}" if 'gate_value' in metrics else ""
            logger.info(f"    F1-Macro: {metrics['f1_macro']:.4f}{gate_str}")
    
    all_results['FULL'] = full_results
    
    # 3. Run WITHOUT_X ablations for each group using PMI_Ablation_Dataset with masking
    for group_name, mask_indices in feature_groups.items():
        ablation_name = f'WITHOUT_{group_name}'
        logger.info(f"\n" + "="*60)
        logger.info(f"Running: {ablation_name}")
        logger.info(f"Masking {len(mask_indices)} features: {mask_indices}")
        logger.info("Using PMI_Ablation_Dataset with HP_PMI hyperparameters")
        logger.info("="*60)
        
        group_results = []
        
        # Create masked dataset for this group
        masked_train = PMI_Ablation_Dataset(
            mode="train", target_type="step-set",
            mask_indices=mask_indices,
            **PMI_CONFIG
        )
        masked_valid = PMI_Ablation_Dataset(
            mode="valid", target_type="step-set",
            mask_indices=mask_indices,
            **PMI_CONFIG
        )
        masked_combined = ConcatDataset([masked_train, masked_valid])
        
        # Use same pre-generated splits
        for repeat_idx in range(N_REPEATS):
            for fold_idx, (train_idx, val_idx) in enumerate(all_splits[repeat_idx]):
                logger.info(f"  {ablation_name} - Repeat {repeat_idx+1}/{N_REPEATS}, Fold {fold_idx+1}/{N_FOLDS}")
                
                metrics = train_ablation_fold(
                    repeat_idx, fold_idx, train_idx, val_idx, 
                    masked_combined, labels_augmented, ablation_name, HP_PMI
                )
                group_results.append(metrics)
                logger.info(f"    F1-Macro: {metrics['f1_macro']:.4f}")
        
        all_results[ablation_name] = group_results
    
    # Save raw results with metadata
    results_with_metadata = {
        'config': {
            'n_folds': N_FOLDS,
            'n_repeats': N_REPEATS,
            'seed': SEED,
            'stratification': 'MultilabelStratifiedKFold',
            'masking_strategy': 'mean',
            'threshold_tuning': 'inner_split_per_fold',
            'hp_geom': HP_GEOM,
            'hp_pmi': HP_PMI,
            'datasets': {
                'geometry_only': 'TKMS_Process_Dataset',
                'full_and_ablations': 'PMI_Ablation_Dataset'
            }
        },
        'feature_groups': feature_groups,
        'results': all_results
    }
    
    with open(OUTPUT_DIR / 'ablation_raw_results.json', 'w') as f:
        json.dump(results_with_metadata, f, indent=4, default=float)
    
    # Analyze results with proper paired comparisons
    analysis_results = analyze_paired_results(all_results, feature_groups)
    
    return all_results, analysis_results


def analyze_paired_results(results, feature_groups):
    """Analyze WITH proper paired comparisons including geometry baseline"""
    
    # Extract results for all conditions
    geom_results = results['GEOMETRY_ONLY']
    full_results = results['FULL']
    
    geom_f1_per_fold = np.array([r['f1_macro'] for r in geom_results])
    full_f1_per_fold = np.array([r['f1_macro'] for r in full_results])
    
    logger.info("\n" + "="*80)
    logger.info("ABLATION ANALYSIS SUMMARY (Paired Comparisons)")
    logger.info("="*80)
    
    # Report both baselines
    logger.info(f"\nGEOMETRY_ONLY Performance (this run):")
    logger.info(f"  Mean F1-Macro: {geom_f1_per_fold.mean():.4f} ± {geom_f1_per_fold.std():.4f}")
    logger.info(f"  Min/Max: [{geom_f1_per_fold.min():.4f}, {geom_f1_per_fold.max():.4f}]")
    
    logger.info(f"\nFULL PMI Performance (this run):")
    logger.info(f"  Mean F1-Macro: {full_f1_per_fold.mean():.4f} ± {full_f1_per_fold.std():.4f}")
    logger.info(f"  Min/Max: [{full_f1_per_fold.min():.4f}, {full_f1_per_fold.max():.4f}]")
    
    # Calculate total PMI gain from this run's actual results
    total_pmi_gain_measured = full_f1_per_fold.mean() - geom_f1_per_fold.mean()
    
    # Ensure we have a positive gain for relative calculations
    total_pmi_gain = max(total_pmi_gain_measured, 1e-8)
    
    logger.info(f"\nTotal PMI gain (measured in this run):")
    logger.info(f"  Gain: {total_pmi_gain_measured:.4f}")
    logger.info(f"  Geometry mean: {geom_f1_per_fold.mean():.4f}")
    logger.info(f"  Full PMI mean: {full_f1_per_fold.mean():.4f}")
    
    # Analyze each group with paired comparisons
    analysis_df = []
    
    for group_name in feature_groups.keys():
        without_results = results[f'WITHOUT_{group_name}']
        without_f1_per_fold = np.array([r['f1_macro'] for r in without_results])
        
        # Paired differences (same fold indices)
        paired_deltas = full_f1_per_fold - without_f1_per_fold
        
        # Bootstrap CI on paired differences
        drop_mean, drop_ci_low, drop_ci_high = calculate_bootstrap_ci(paired_deltas)
        
        # Wilcoxon signed-rank test (one-sided: drop > 0)
        if len(paired_deltas) >= 5:
            stat_wilcox, p_wilcox = wilcoxon(paired_deltas, alternative='greater')
        else:
            stat_wilcox, p_wilcox = np.nan, np.nan
        
        # Per-class analysis (also paired)
        per_class_drops = {}
        for class_name in CLASS_NAMES:
            class_key = f'f1_{class_name.lower()}'
            full_class = np.array([r[class_key] for r in full_results])
            without_class = np.array([r[class_key] for r in without_results])
            per_class_drops[class_name] = (full_class - without_class).mean()
        
        # Gate value analysis if available
        gate_values = [r.get('gate_value', np.nan) for r in without_results]
        mean_gate = np.nanmean(gate_values) if any(~np.isnan(gate_values)) else np.nan
        
        analysis_df.append({
            'Group': group_name,
            'F1_WITHOUT_mean': without_f1_per_fold.mean(),
            'F1_WITHOUT_std': without_f1_per_fold.std(),
            'Performance_Drop': drop_mean,
            'Drop_CI_Low': drop_ci_low,
            'Drop_CI_High': drop_ci_high,
            'Wilcoxon_statistic': stat_wilcox,
            'Wilcoxon_p': p_wilcox,
            'Significant': p_wilcox < ALPHA if not np.isnan(p_wilcox) else False,
            'Relative_Importance_%': (drop_mean / total_pmi_gain * 100) if total_pmi_gain > 0 else 0,
            'Drop_Bohren': per_class_drops['Bohren'],
            'Drop_Drehen': per_class_drops['Drehen'],
            'Drop_Fraesen': per_class_drops['Fräsen'],
            'Mean_Gate': mean_gate
        })
        
        logger.info(f"\n{group_name}:")
        logger.info(f"  F1 WITHOUT: {without_f1_per_fold.mean():.4f} ± {without_f1_per_fold.std():.4f}")
        logger.info(f"  Paired Drop: {drop_mean:.4f} [{drop_ci_low:.4f}, {drop_ci_high:.4f}]")
        logger.info(f"  Wilcoxon p: {p_wilcox:.4f} {'✓' if p_wilcox < ALPHA else '✗'}")
        logger.info(f"  Importance: {drop_mean/total_pmi_gain*100:.1f}% of total PMI gain")
        logger.info(f"  Per-class drops: B={per_class_drops['Bohren']:.3f}, "
                   f"D={per_class_drops['Drehen']:.3f}, F={per_class_drops['Fräsen']:.3f}")
        if not np.isnan(mean_gate):
            logger.info(f"  Mean gate value: {mean_gate:.3f}")
    
    # Create DataFrame and sort by importance
    df = pd.DataFrame(analysis_df).sort_values('Performance_Drop', ascending=False)
    df.to_csv(OUTPUT_DIR / 'ablation_analysis_paired.csv', index=False)
    
    # Create enhanced visualizations including geometry baseline
    create_enhanced_ablation_plots(df, geom_f1_per_fold, full_f1_per_fold, results, feature_groups, OUTPUT_DIR)
    
    # Print final ranking
    logger.info("\n" + "="*80)
    logger.info("FEATURE GROUP IMPORTANCE RANKING (Paired Analysis)")
    logger.info("="*80)
    for idx, row in df.iterrows():
        sig_marker = "✓" if row['Significant'] else ""
        logger.info(f"{idx+1}. {row['Group']:25s}: "
                   f"Drop={row['Performance_Drop']:.4f} "
                   f"({row['Relative_Importance_%']:.1f}%) "
                   f"p={row['Wilcoxon_p']:.3f} {sig_marker}")
    
    # Calculate and report gate correlation if available
    if not df['Mean_Gate'].isna().all():
        from scipy.stats import pearsonr
        valid_gates = df.dropna(subset=['Mean_Gate'])
        if len(valid_gates) > 2:
            corr, p_corr = pearsonr(valid_gates['Performance_Drop'], valid_gates['Mean_Gate'])
            logger.info(f"\nGate-Importance Correlation: r={corr:.3f}, p={p_corr:.3f}")
    
    return df


def create_enhanced_ablation_plots(df, geom_f1, full_f1, results, feature_groups, output_dir):
    """Create publication-ready ablation plots including geometry baseline"""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Sort by importance
    df_sorted = df.sort_values('Performance_Drop', ascending=False)
    
    # 1. Performance comparison with all baselines
    ax = axes[0, 0]
    
    # Prepare data for boxplot
    bp_data = [geom_f1, full_f1]
    bp_labels = ['Geometry\nOnly', 'Full\nPMI']
    
    for group_name in df_sorted['Group']:
        without_f1 = np.array([r['f1_macro'] for r in results[f'WITHOUT_{group_name}']])
        bp_data.append(without_f1)
        bp_labels.append(f'WITHOUT\n{group_name[:8]}')
    
    bp = ax.boxplot(bp_data[:8], labels=bp_labels[:8], patch_artist=True)  # Show first 8 for space
    bp['boxes'][0].set_facecolor('lightblue')  # Geometry
    bp['boxes'][1].set_facecolor('darkgreen')  # Full PMI
    for i in range(2, len(bp['boxes'])):
        bp['boxes'][i].set_facecolor('coral')  # WITHOUT variants
    
    ax.set_ylabel('F1-Macro Score')
    ax.set_title('Performance Across All Conditions')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Performance drop with CI and significance
    ax = axes[0, 1]
    x_pos = np.arange(len(df_sorted))
    bars = ax.bar(x_pos, df_sorted['Performance_Drop'], alpha=0.7, color='coral')
    
    # Color significant bars differently
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        if row['Significant']:
            bars[i].set_color('darkred')
    
    ax.errorbar(x_pos, df_sorted['Performance_Drop'], 
                yerr=[df_sorted['Performance_Drop'] - df_sorted['Drop_CI_Low'],
                      df_sorted['Drop_CI_High'] - df_sorted['Performance_Drop']],
                fmt='none', color='black', capsize=3)
    
    # Add significance stars
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        if row['Wilcoxon_p'] < 0.001:
            ax.text(i, row['Drop_CI_High'] + 0.002, '***', ha='center', fontsize=10)
        elif row['Wilcoxon_p'] < 0.01:
            ax.text(i, row['Drop_CI_High'] + 0.002, '**', ha='center', fontsize=10)
        elif row['Wilcoxon_p'] < 0.05:
            ax.text(i, row['Drop_CI_High'] + 0.002, '*', ha='center', fontsize=10)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_sorted['Group'], rotation=45, ha='right')
    ax.set_ylabel('F1-Macro Drop')
    ax.set_title('Performance Drop When Removing Each PMI Group')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Per-class impact heatmap
    ax = axes[0, 2]
    class_drops = df_sorted[['Group', 'Drop_Bohren', 'Drop_Drehen', 'Drop_Fraesen']].set_index('Group')
    sns.heatmap(class_drops.T, annot=True, fmt='.3f', cmap='Reds', 
                cbar_kws={'label': 'F1 Drop'}, ax=ax, vmin=0)
    ax.set_xlabel('PMI Feature Group')
    ax.set_ylabel('Manufacturing Process')
    ax.set_title('Per-Class Performance Drop')
    
    # 4. Relative importance pie chart
    ax = axes[1, 0]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(df_sorted)))
    wedges, texts, autotexts = ax.pie(df_sorted['Relative_Importance_%'], 
                                       labels=df_sorted['Group'], 
                                       autopct='%1.1f%%',
                                       colors=colors,
                                       startangle=90)
    ax.set_title('Relative Importance of PMI Groups')
    
    # 5. Total gain decomposition
    ax = axes[1, 1]
    
    # Calculate cumulative drops
    sorted_drops = df_sorted['Performance_Drop'].values
    cumulative = np.cumsum(sorted_drops)
    total_gain = full_f1.mean() - geom_f1.mean()
    
    x = np.arange(len(df_sorted))
    ax.bar(x, sorted_drops, alpha=0.7, label='Individual drop')
    ax.plot(x, cumulative, 'r-o', label='Cumulative drop')
    ax.axhline(total_gain, color='green', linestyle='--', label=f'Total PMI gain: {total_gain:.4f}')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted['Group'], rotation=45, ha='right')
    ax.set_ylabel('F1-Macro')
    ax.set_title('Cumulative Feature Group Contributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Statistical summary table
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')
    
    # Include baseline info in table
    summary_data = [
        ['BASELINE', '', '', '', ''],
        ['Geometry', f"{geom_f1.mean():.3f}", '-', '-', '-'],
        ['Full PMI', f"{full_f1.mean():.3f}", '-', '-', '-'],
        ['Total Gain', f"{total_gain:.3f}", '-', '-', '-'],
        ['', '', '', '', ''],
        ['ABLATIONS', '', '', '', '']
    ]
    
    for _, row in df_sorted.head(4).iterrows():
        sig_marker = '✓' if row['Significant'] else '✗'
        summary_data.append([
            row['Group'][:12],
            f"{row['Performance_Drop']:.3f}",
            f"[{row['Drop_CI_Low']:.3f}, {row['Drop_CI_High']:.3f}]",
            f"{row['Wilcoxon_p']:.3f}",
            sig_marker
        ])
    
    table = ax.table(cellText=summary_data,
                     colLabels=['Condition', 'F1/Drop', '95% CI', 'p-value', 'Sig'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.5)
    
    ax.set_title('Statistical Summary', fontsize=11, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_results_complete.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\nPlots saved to {output_dir / 'ablation_results_complete.png'}")


if __name__ == "__main__":
    try:
        results, analysis = run_without_x_ablations()
        logger.info(f"\n✓ Ablation study completed successfully!")
        logger.info(f"✓ Results saved in: {OUTPUT_DIR}")
        
        # Print key findings
        logger.info("\n" + "="*80)
        logger.info("KEY FINDINGS")
        logger.info("="*80)
        
        top3 = analysis.nlargest(3, 'Performance_Drop')
        logger.info("\nMost important PMI groups (by paired drop):")
        for idx, row in top3.iterrows():
            sig = "✓" if row['Significant'] else ""
            logger.info(f"  • {row['Group']}: {row['Relative_Importance_%']:.1f}% "
                       f"(p={row['Wilcoxon_p']:.3f}) {sig}")
        
        logger.info("\nGreatest impact on Drehen/Fräsen discrimination:")
        df_fraesen = analysis.nlargest(3, 'Drop_Fraesen')
        for idx, row in df_fraesen.iterrows():
            logger.info(f"  • {row['Group']}: Fräsen drop = {row['Drop_Fraesen']:.3f}")
        
        # Summary statistics
        significant_groups = analysis[analysis['Significant']]
        logger.info(f"\nStatistically significant groups: {len(significant_groups)}/{len(analysis)}")
        if len(significant_groups) > 0:
            logger.info("Significant groups: " + 
                       ", ".join(significant_groups['Group'].tolist()))
            
    except Exception as e:
        logger.error(f"Error during ablation study: {str(e)}", exc_info=True)
        raise