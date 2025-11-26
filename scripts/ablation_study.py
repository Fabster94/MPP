#!/usr/bin/env python3
"""
PMI Feature Ablation Study with Feature Removal and KEY_PMI Analysis
Includes skip logic for already completed experiments
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging
import warnings

import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

from sklearn.metrics import f1_score, jaccard_score, accuracy_score
from torch.utils.data import DataLoader, Subset, ConcatDataset
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings('ignore', category=UserWarning)

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from mpp.ml.models.classifier.unified_process_classifier import UnifiedProcessClassifier
from mpp.ml.datasets.tkms import TKMS_Process_Dataset
from mpp.ml.datasets.tkms_pmi import TKMS_PMI_Dataset

# ========== CONFIGURATION ==========
N_FOLDS = 5
N_REPEATS = 5
SEED = 42
BATCH_SIZE = 85
MAX_EPOCHS = 100
PATIENCE = 20
CLASS_NAMES = ["Bohren", "Drehen", "Fräsen"]
NUM_WORKERS = 0  # Faster data loading with multiple workers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_BOOTSTRAP = 5000
ALPHA = 0.05

# Path to your existing results
PREVIOUS_RESULTS_PATH = Path("/workspace/masterthesis_cadtoplan_fabian_heinze/ablation_results_removal/20251020_164726/ablation_raw_results.json")

OUTPUT_DIR = Path("ablation_results_removal") / datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(OUTPUT_DIR / 'ablation_log.txt')),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Best hyperparameters from main CV - FIXED VERSION
HP_GEOM = {
    "dropout": 0.224,
    "lr": 0.000326,
    "embed_dim": 128,
    "num_layers": 2,
    "num_heads": 16,
    "weight_decay": 0.000374,
    "use_pmi": False,
    "pmi_dim": 30,
    "initial_gate": 0.2,
    "modality_dropout": 0.0
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
    "modality_dropout": 0.0 #0.206
}

PMI_CONFIG = {}


def get_pmi_feature_groups():
    """Define PMI feature groups with CORRECT indices matching CSV column order"""
    
    feature_groups = {
        'dimensions': [0, 1, 2, 3, 4],  # 5 features
        'fits': [5, 6],  # 2 features
        'dimensional_tolerances': [7, 8, 9, 10, 11, 12, 13, 14, 24],  # 9 features
        'surface_finish': [15, 16, 17, 18],  # 4 features
        'geometric_tolerances': [19, 20, 21, 22, 23, 27, 28, 29],  # 8 features
        'datums': [25, 26]  # 2 features
    }
    
    # Verify all 30 features are covered
    all_indices = []
    for indices in feature_groups.values():
        all_indices.extend(indices)
    assert len(set(all_indices)) == 30, f"Feature mapping error: {len(set(all_indices))} != 30"
    assert set(all_indices) == set(range(30)), "Not all indices 0-29 covered!"
    
    logger.info("Feature groups defined (CORRECTED):")
    for group, indices in feature_groups.items():
        logger.info(f"  {group}: {len(indices)} features (indices {min(indices)}-{max(indices)})")
    
    return feature_groups


class PMI_Removal_Dataset(TKMS_PMI_Dataset):
    """Extended dataset with actual feature removal (not masking)"""
    
    def __init__(self, remove_indices=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remove_indices = remove_indices if remove_indices is not None else []
        
        # Create keep mask (inverse of remove)
        self.keep_indices = [i for i in range(30) if i not in self.remove_indices]
        self.output_dim = len(self.keep_indices)
        
        if len(self.remove_indices) > 0:
            logger.debug(f"Removing {len(self.remove_indices)} features, keeping {self.output_dim}")
    
    def __getitem__(self, idx):
        # Get the original item from parent class
        (vecset, pmi_original), label = super().__getitem__(idx)
        
        # Remove specified features
        if len(self.remove_indices) > 0:
            pmi_reduced = pmi_original[self.keep_indices].float()
        else:
            pmi_reduced = pmi_original.float()
        
        return (vecset, pmi_reduced), label


def calculate_bootstrap_ci(data, n_bootstrap=N_BOOTSTRAP, alpha=ALPHA, seed=SEED):
    """Calculate bootstrap confidence intervals with fixed RNG for reproducibility"""
    if len(data) < 2:
        return np.mean(data), np.nan, np.nan
    
    rng = np.random.default_rng(seed)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_means, (alpha/2) * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return np.mean(data), lower, upper


def holm_adjust(pvals, alpha=ALPHA):
    """Holm-Bonferroni multiple testing correction"""
    p = np.array(pvals, dtype=float)
    m = np.sum(~np.isnan(p))
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    adj = np.full_like(p, np.nan, dtype=float)
    running_max = 0.0
    k = 0
    for idx in order:
        if np.isnan(p[idx]): 
            continue
        adj_val = (m - k) * p[idx]
        running_max = max(running_max, adj_val)
        adj[idx] = min(running_max, 1.0)
        k += 1
    reject = adj < alpha
    return adj, reject


def load_aligned_datasets():
    """Load geometry and PMI datasets ensuring sample alignment"""
    train_geom = TKMS_Process_Dataset(mode="train", target_type="step-set")
    valid_geom = TKMS_Process_Dataset(mode="valid", target_type="step-set")
    
    train_pmi = TKMS_PMI_Dataset(mode="train", target_type="step-set", **PMI_CONFIG)
    valid_pmi = TKMS_PMI_Dataset(mode="valid", target_type="step-set", **PMI_CONFIG)
    
    # Verify sample alignment
    assert train_pmi.samples == train_geom.samples, "Train sample order mismatch!"
    assert valid_pmi.samples == valid_geom.samples, "Valid sample order mismatch!"
    
    all_sample_ids = train_pmi.samples + valid_pmi.samples
    
    # Only geometry dataset is used for GEOMETRY_ONLY baseline
    dataset_geom = ConcatDataset([train_geom, valid_geom])
    
    # Get labels for stratification
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
    interactions[:, 0] = labels_int[:, 0] & labels_int[:, 1]
    interactions[:, 1] = labels_int[:, 0] & labels_int[:, 2]
    interactions[:, 2] = labels_int[:, 1] & labels_int[:, 2]
    labels_augmented = np.hstack([labels_int, interactions])
    
    logger.info(f"Total samples: {len(labels)}")
    logger.info(f"Label distribution: {labels.sum(axis=0)} ({CLASS_NAMES})")
    
    return dataset_geom, labels, labels_augmented, all_sample_ids


def generate_all_splits(labels_augmented, n_repeats=N_REPEATS, n_folds=N_FOLDS, seed=SEED):
    """Pre-generate all CV splits to ensure perfect pairing across ablations"""
    all_splits = {}
    
    for repeat_idx in range(n_repeats):
        repeat_seed = seed + repeat_idx * 1000
        splitter = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=repeat_seed)
        splits = list(splitter.split(np.zeros(len(labels_augmented)), labels_augmented))
        all_splits[repeat_idx] = splits
        logger.debug(f"Generated splits for repeat {repeat_idx}: {n_folds} folds")
    
    return all_splits


def train_ablation_fold(repeat_idx, fold_idx, train_idx, val_idx, dataset, 
                        ablation_name, hyperparams, pmi_dim=30):
    """Train single fold with specified configuration"""
    fold_seed = SEED + repeat_idx * 1000 + fold_idx * 100
    seed_everything(fold_seed, workers=True)
    
    use_pmi = (ablation_name != 'GEOMETRY_ONLY')
    
    # Log more details about current fold
    logger.info(f"    Fold details: train={len(train_idx)} samples, val={len(val_idx)} samples")
    logger.info(f"    Model config: {'PMI' if use_pmi else 'Geometry-only'}, pmi_dim={pmi_dim if use_pmi else 'N/A'}")
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    # Persistent workers for better performance (only if NUM_WORKERS > 0)
    persistent = NUM_WORKERS > 0
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=2 if persistent else None
    )
    
    # Build model parameters correctly
    model_params = {
        'num_classes': len(CLASS_NAMES),
        **hyperparams  # This already includes use_pmi and all other params
    }
    
    # Override pmi_dim for reduced models (when features are removed)
    if use_pmi and pmi_dim != 30:
        model_params['pmi_dim'] = pmi_dim
    
    model = UnifiedProcessClassifier(**model_params).to(DEVICE)
    
    checkpoint_dir = OUTPUT_DIR / f"repeat_{repeat_idx}" / f"fold_{fold_idx}" / ablation_name.lower()
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        mode='min',
        verbose=True  # Show when early stopping triggers
    )
    
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,  # Show progress bar
        logger=False,  # Keep False to avoid TensorBoard overhead
        deterministic=True,
        enable_model_summary=False  # Reduce clutter
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model with correct parameters
    checkpoint_params = {
        'num_classes': len(CLASS_NAMES),
        **hyperparams
    }
    
    if use_pmi and pmi_dim != 30:
        checkpoint_params['pmi_dim'] = pmi_dim
    
    best_model = UnifiedProcessClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        **checkpoint_params
    ).to(DEVICE)
    
    # Fixed threshold at 0.5 (no threshold tuning)
    threshold = 0.5
    
    # Evaluate on validation set
    all_preds = []
    all_labels = []
    all_probs = []
    
    best_model.eval()
    with torch.no_grad():
        for batch in val_loader:
            if use_pmi:
                (inputs, pmi), labels = batch
                outputs = best_model(inputs.to(DEVICE), pmi.to(DEVICE))
            else:
                inputs, labels = batch
                outputs = best_model(inputs.to(DEVICE))
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            all_preds.append(preds)
            all_labels.append(labels.numpy())
            all_probs.append(probs)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    jaccard = jaccard_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    
    metrics = {
        'f1_macro': f1_macro,
        'f1_bohren': f1_per_class[0],
        'f1_drehen': f1_per_class[1],
        'f1_fraesen': f1_per_class[2],
        'jaccard': jaccard,
        'accuracy': accuracy,
        'epochs_trained': trainer.current_epoch,
        'threshold': threshold
    }
    
    # Gate value for PMI models (robust handling)
    if use_pmi and hasattr(best_model, 'gate'):
        w = getattr(best_model.gate, 'weight', None)
        if w is not None:
            metrics['gate_value'] = torch.sigmoid(w).mean().item()
    
    # Log detailed results for this fold
    logger.info(f"    Results: F1-Macro={f1_macro:.4f}, Epochs={trainer.current_epoch}")
    logger.info(f"    Per-class F1: Bohren={f1_per_class[0]:.3f}, Drehen={f1_per_class[1]:.3f}, Fräsen={f1_per_class[2]:.3f}")
    
    return metrics


def run_feature_removal_ablations():
    """Run feature removal ablations with repeated CV"""
    logger.info("\n" + "="*80)
    logger.info("PMI FEATURE ABLATION STUDY - FEATURE REMOVAL ANALYSIS")
    logger.info("="*80)
    logger.info(f"Configuration: {N_REPEATS}×{N_FOLDS} = {N_REPEATS*N_FOLDS} total folds")
    logger.info(f"Batch size: {BATCH_SIZE}, Max epochs: {MAX_EPOCHS}, Patience: {PATIENCE}")
    logger.info(f"Device: {DEVICE}")
    logger.info("="*80)
    
    feature_groups = get_pmi_feature_groups()
    
    dataset_geom, labels, labels_augmented, sample_ids = load_aligned_datasets()
    
    logger.info("\nPre-generating CV splits for paired comparisons...")
    all_splits = generate_all_splits(labels_augmented)
    
    # Check if previous results exist to skip completed experiments
    if PREVIOUS_RESULTS_PATH.exists():
        logger.info(f"\n⚠️  Loading existing results from: {PREVIOUS_RESULTS_PATH}")
        with open(PREVIOUS_RESULTS_PATH, 'r') as f:
            existing_data = json.load(f)
            all_results = existing_data.get('results', {})
        logger.info(f"   Loaded results for: {list(all_results.keys())}")
        logger.info(f"   Number of folds per experiment: {len(all_results.get('GEOMETRY_ONLY', []))}")
    else:
        logger.info(f"\n⚠️  No previous results found at: {PREVIOUS_RESULTS_PATH}")
        logger.info("   Starting fresh with all experiments...")
        all_results = {}
    
    # Save splits with indices
    splits_for_save = {}
    for repeat_idx, splits in all_splits.items():
        splits_for_save[f"repeat_{repeat_idx}"] = [
            {"train": train_idx.tolist(), "val": val_idx.tolist()} 
            for train_idx, val_idx in splits
        ]
    with open(OUTPUT_DIR / 'cv_splits.json', 'w') as f:
        json.dump(splits_for_save, f, indent=2)
    
    # Save splits with sample IDs for auditability
    splits_for_save_ids = {}
    for repeat_idx, splits in all_splits.items():
        key = f"repeat_{repeat_idx}"
        splits_for_save_ids[key] = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            splits_for_save_ids[key].append({
                "fold": fold_idx,
                "train_ids": [sample_ids[i] for i in train_idx],
                "val_ids": [sample_ids[i] for i in val_idx],
                "train_count": len(train_idx),
                "val_count": len(val_idx)
            })
    with open(OUTPUT_DIR / 'cv_splits_ids.json', 'w') as f:
        json.dump(splits_for_save_ids, f, indent=2)
    
    logger.info(f"Saved CV splits to {OUTPUT_DIR / 'cv_splits.json'}")
    logger.info(f"Saved CV split IDs to {OUTPUT_DIR / 'cv_splits_ids.json'}")
    
    # 1. GEOMETRY_ONLY baseline - SKIP if already done
    if 'GEOMETRY_ONLY' not in all_results:
        logger.info("\n" + "="*60)
        logger.info("Running: GEOMETRY_ONLY (Baseline - No PMI)")
        logger.info("Using TKMS_Process_Dataset with HP_GEOM hyperparameters")
        logger.info("="*60)
        
        geom_results = []
        for repeat_idx in range(N_REPEATS):
            for fold_idx, (train_idx, val_idx) in enumerate(all_splits[repeat_idx]):
                logger.info(f"\n  GEOMETRY_ONLY - Repeat {repeat_idx+1}/{N_REPEATS}, Fold {fold_idx+1}/{N_FOLDS}")
                metrics = train_ablation_fold(
                    repeat_idx, fold_idx, train_idx, val_idx, 
                    dataset_geom, 'GEOMETRY_ONLY', HP_GEOM
                )
                geom_results.append(metrics)
                logger.info(f"    → F1-Macro: {metrics['f1_macro']:.4f}, Epochs: {metrics['epochs_trained']}")
        
        all_results['GEOMETRY_ONLY'] = geom_results
        
        # Print summary for GEOMETRY_ONLY
        geom_f1_array = np.array([r['f1_macro'] for r in geom_results])
        logger.info(f"\n  GEOMETRY_ONLY Summary:")
        logger.info(f"    Mean F1: {geom_f1_array.mean():.4f} ± {geom_f1_array.std():.4f}")
        logger.info(f"    Min/Max: [{geom_f1_array.min():.4f}, {geom_f1_array.max():.4f}]")
    else:
        logger.info("\n✓ Skipping GEOMETRY_ONLY - already completed")
    
    # 2. FULL control - SKIP if already done
    if 'FULL' not in all_results:
        logger.info("\n" + "="*60)
        logger.info("Running: FULL (Control - All PMI Features)")
        logger.info("Using PMI_Removal_Dataset with HP_PMI hyperparameters (no removal)")
        logger.info("="*60)
        
        full_results = []
        full_dataset = PMI_Removal_Dataset(
            mode="train", target_type="step-set",
            remove_indices=[],  # No features removed
            **PMI_CONFIG
        )
        valid_dataset = PMI_Removal_Dataset(
            mode="valid", target_type="step-set",
            remove_indices=[],
            **PMI_CONFIG
        )
        full_combined = ConcatDataset([full_dataset, valid_dataset])
        
        for repeat_idx in range(N_REPEATS):
            for fold_idx, (train_idx, val_idx) in enumerate(all_splits[repeat_idx]):
                logger.info(f"\n  FULL - Repeat {repeat_idx+1}/{N_REPEATS}, Fold {fold_idx+1}/{N_FOLDS}")
                metrics = train_ablation_fold(
                    repeat_idx, fold_idx, train_idx, val_idx, 
                    full_combined, 'FULL', HP_PMI, pmi_dim=30
                )
                full_results.append(metrics)
                gate_str = f", Gate: {metrics.get('gate_value', 'N/A'):.3f}" if 'gate_value' in metrics else ""
                logger.info(f"    → F1-Macro: {metrics['f1_macro']:.4f}{gate_str}, Epochs: {metrics['epochs_trained']}")
        
        all_results['FULL'] = full_results
        
        # Print summary for FULL
        full_f1_array = np.array([r['f1_macro'] for r in full_results])
        logger.info(f"\n  FULL PMI Summary:")
        logger.info(f"    Mean F1: {full_f1_array.mean():.4f} ± {full_f1_array.std():.4f}")
        logger.info(f"    Min/Max: [{full_f1_array.min():.4f}, {full_f1_array.max():.4f}]")
        logger.info(f"    Improvement over GEOMETRY: +{(full_f1_array.mean() - np.array([r['f1_macro'] for r in all_results['GEOMETRY_ONLY']]).mean()):.4f}")
    else:
        logger.info("\n✓ Skipping FULL - already completed")
    
    # 3. WITHOUT_X ablations - SKIP already completed ones
    for group_name, remove_indices in feature_groups.items():
        ablation_name = f'WITHOUT_{group_name}'
        
        if ablation_name in all_results:
            logger.info(f"\n✓ Skipping {ablation_name} - already completed")
            continue
            
        logger.info(f"\n" + "="*60)
        logger.info(f"Running: {ablation_name}")
        logger.info(f"Removing {len(remove_indices)} features: indices {remove_indices}")
        logger.info(f"Resulting PMI dimension: {30 - len(remove_indices)}")
        logger.info("Using PMI_Removal_Dataset with HP_PMI hyperparameters")
        logger.info("="*60)
        
        group_results = []
        
        # Create dataset with features removed
        removed_train = PMI_Removal_Dataset(
            mode="train", target_type="step-set",
            remove_indices=remove_indices,
            **PMI_CONFIG
        )
        removed_valid = PMI_Removal_Dataset(
            mode="valid", target_type="step-set",
            remove_indices=remove_indices,
            **PMI_CONFIG
        )
        removed_combined = ConcatDataset([removed_train, removed_valid])
        
        reduced_pmi_dim = 30 - len(remove_indices)
        
        for repeat_idx in range(N_REPEATS):
            for fold_idx, (train_idx, val_idx) in enumerate(all_splits[repeat_idx]):
                logger.info(f"\n  {ablation_name} - Repeat {repeat_idx+1}/{N_REPEATS}, Fold {fold_idx+1}/{N_FOLDS}")
                metrics = train_ablation_fold(
                    repeat_idx, fold_idx, train_idx, val_idx, 
                    removed_combined, ablation_name, HP_PMI, 
                    pmi_dim=reduced_pmi_dim
                )
                group_results.append(metrics)
                logger.info(f"    → F1-Macro: {metrics['f1_macro']:.4f}, Epochs: {metrics['epochs_trained']}")
        
        all_results[ablation_name] = group_results
        
        # Print summary for this ablation
        without_f1_array = np.array([r['f1_macro'] for r in group_results])
        logger.info(f"\n  {ablation_name} Summary:")
        logger.info(f"    Mean F1: {without_f1_array.mean():.4f} ± {without_f1_array.std():.4f}")
        logger.info(f"    Performance drop from FULL: -{(np.array([r['f1_macro'] for r in all_results['FULL']]).mean() - without_f1_array.mean()):.4f}")
    
    # 4. ONLY_KEY_PMI - NEW EXPERIMENT with only critical features
    if 'ONLY_KEY_PMI' not in all_results:
        logger.info("\n" + "="*60)
        logger.info("🔑 Running: ONLY_KEY_PMI (Dimensions + Geometric Tolerances Only)")
        logger.info("="*60)
        
        # Define key features to KEEP (dimensions + geometric_tolerances)
        key_features_to_keep = [
            0, 1, 2, 3, 4,  # dimensions (5 features)
            19, 20, 21, 22, 23, 27, 28, 29  # geometric_tolerances (8 features)
        ]
        
        # Convert to remove_indices (inverse logic for PMI_Removal_Dataset)
        all_indices = set(range(30))
        key_indices_set = set(key_features_to_keep)
        remove_indices_for_key = sorted(list(all_indices - key_indices_set))
        
        logger.info(f"  Keeping {len(key_features_to_keep)} features: {sorted(key_features_to_keep)}")
        logger.info(f"  Removing {len(remove_indices_for_key)} features: {remove_indices_for_key}")
        logger.info(f"  Feature reduction: 30 → {len(key_features_to_keep)} (43% of original)")
        logger.info(f"  Expected to capture ~76% of PMI benefit (based on ablation results)")
        
        key_results = []
        
        # Create dataset with only key features
        key_train = PMI_Removal_Dataset(
            mode="train", target_type="step-set",
            remove_indices=remove_indices_for_key,
            **PMI_CONFIG
        )
        key_valid = PMI_Removal_Dataset(
            mode="valid", target_type="step-set",
            remove_indices=remove_indices_for_key,
            **PMI_CONFIG
        )
        key_combined = ConcatDataset([key_train, key_valid])
        
        key_pmi_dim = len(key_features_to_keep)  # 13
        
        for repeat_idx in range(N_REPEATS):
            for fold_idx, (train_idx, val_idx) in enumerate(all_splits[repeat_idx]):
                logger.info(f"\n  ONLY_KEY_PMI - Repeat {repeat_idx+1}/{N_REPEATS}, Fold {fold_idx+1}/{N_FOLDS}")
                metrics = train_ablation_fold(
                    repeat_idx, fold_idx, train_idx, val_idx,
                    key_combined, 'ONLY_KEY_PMI', HP_PMI,
                    pmi_dim=key_pmi_dim
                )
                key_results.append(metrics)
                gate_str = f", Gate: {metrics.get('gate_value', 'N/A'):.3f}" if 'gate_value' in metrics else ""
                logger.info(f"    → F1-Macro: {metrics['f1_macro']:.4f}{gate_str}, Epochs: {metrics['epochs_trained']}")
        
        all_results['ONLY_KEY_PMI'] = key_results
        
        # Print summary and comparison
        key_f1_array = np.array([r['f1_macro'] for r in key_results])
        logger.info(f"\n  🔑 ONLY_KEY_PMI Summary:")
        logger.info(f"    Mean F1: {key_f1_array.mean():.4f} ± {key_f1_array.std():.4f}")
        logger.info(f"    Min/Max: [{key_f1_array.min():.4f}, {key_f1_array.max():.4f}]")
        
        # Compare with baselines if available
        if 'GEOMETRY_ONLY' in all_results and 'FULL' in all_results:
            geom_mean = np.mean([r['f1_macro'] for r in all_results['GEOMETRY_ONLY']])
            full_mean = np.mean([r['f1_macro'] for r in all_results['FULL']])
            key_mean = key_f1_array.mean()
            
            total_gain = full_mean - geom_mean
            key_gain = key_mean - geom_mean
            efficiency = (key_gain / total_gain) * 100 if total_gain > 0 else 0
            
            logger.info(f"\n  📊 Feature Efficiency Analysis:")
            logger.info(f"    GEOMETRY_ONLY:  {geom_mean:.4f}")
            logger.info(f"    ONLY_KEY_PMI:   {key_mean:.4f} (+{key_gain:.4f})")
            logger.info(f"    FULL_PMI:       {full_mean:.4f} (+{total_gain:.4f})")
            logger.info(f"    → KEY features capture {efficiency:.1f}% of total PMI benefit")
            logger.info(f"    → Using only 43% of PMI features!")
    else:
        logger.info("\n✓ Skipping ONLY_KEY_PMI - already completed")
    
    # Save results
    results_with_metadata = {
        'config': {
            'n_folds': N_FOLDS,
            'n_repeats': N_REPEATS,
            'seed': SEED,
            'stratification': 'MultilabelStratifiedKFold',
            'ablation_strategy': 'feature_removal',
            'threshold': 0.5,
            'deterministic': True,
            'hp_geom': HP_GEOM,
            'hp_pmi': HP_PMI,
            'datasets': {
                'geometry_only': 'TKMS_Process_Dataset',
                'full_and_ablations': 'PMI_Removal_Dataset'
            }
        },
        'feature_groups': feature_groups,
        'results': all_results
    }
    
    with open(OUTPUT_DIR / 'ablation_raw_results.json', 'w') as f:
        json.dump(results_with_metadata, f, indent=4, default=float)
    
    analysis_results = analyze_paired_results(all_results, feature_groups)
    
    return all_results, analysis_results


def analyze_paired_results(results, feature_groups):
    """Analyze with proper paired comparisons including ONLY_KEY_PMI"""
    
    # Explicit class key mapping (handles umlauts correctly)
    CLASS_KEY_MAP = {
        "Bohren": "f1_bohren",
        "Drehen": "f1_drehen",
        "Fräsen": "f1_fraesen",  # ae not ä
    }
    
    geom_results = results['GEOMETRY_ONLY']
    full_results = results['FULL']
    
    geom_f1_per_fold = np.array([r['f1_macro'] for r in geom_results])
    full_f1_per_fold = np.array([r['f1_macro'] for r in full_results])
    
    logger.info("\n" + "="*80)
    logger.info("ABLATION ANALYSIS SUMMARY (Paired Comparisons)")
    logger.info("="*80)
    
    logger.info(f"\n📊 BASELINE PERFORMANCE:")
    logger.info(f"  GEOMETRY_ONLY: {geom_f1_per_fold.mean():.4f} ± {geom_f1_per_fold.std():.4f}")
    logger.info(f"  FULL PMI:      {full_f1_per_fold.mean():.4f} ± {full_f1_per_fold.std():.4f}")
    
    total_pmi_gain = max(full_f1_per_fold.mean() - geom_f1_per_fold.mean(), 1e-8)
    
    logger.info(f"\n📈 TOTAL PMI CONTRIBUTION: +{total_pmi_gain:.4f}")
    logger.info(f"  Relative improvement: {(total_pmi_gain / geom_f1_per_fold.mean() * 100):.1f}%")
    
    analysis_df = []
    
    # Analyze WITHOUT_X groups
    for group_name in feature_groups.keys():
        without_results = results.get(f'WITHOUT_{group_name}')
        if not without_results:
            continue
            
        without_f1_per_fold = np.array([r['f1_macro'] for r in without_results])
        
        # Paired differences
        paired_deltas = full_f1_per_fold - without_f1_per_fold
        
        # Bootstrap CI
        drop_mean, drop_ci_low, drop_ci_high = calculate_bootstrap_ci(paired_deltas)
        
        # Wilcoxon test
        if len(paired_deltas) >= 5:
            stat_wilcox, p_wilcox = wilcoxon(paired_deltas, alternative='greater')
        else:
            stat_wilcox, p_wilcox = np.nan, np.nan
        
        # Per-class analysis - USE THE MAP
        per_class_drops = {}
        for class_name in CLASS_NAMES:
            class_key = CLASS_KEY_MAP[class_name]
            full_class = np.array([r[class_key] for r in full_results])
            without_class = np.array([r[class_key] for r in without_results])
            per_class_drops[class_name] = (full_class - without_class).mean()
        
        # Gate values
        gate_values = [r.get('gate_value', np.nan) for r in without_results]
        mean_gate = np.nanmean(gate_values) if any(~np.isnan(gate_values)) else np.nan
        
        analysis_df.append({
            'Group': group_name,
            'Type': 'WITHOUT',
            'F1_mean': without_f1_per_fold.mean(),
            'F1_std': without_f1_per_fold.std(),
            'Performance_Drop': drop_mean,
            'Drop_CI_Low': drop_ci_low,
            'Drop_CI_High': drop_ci_high,
            'Relative_Importance_%': (drop_mean / total_pmi_gain) * 100,
            'Wilcoxon_statistic': stat_wilcox,
            'Wilcoxon_p': p_wilcox,
            'Significant': p_wilcox < ALPHA if not np.isnan(p_wilcox) else False,
            'Drop_Bohren': per_class_drops['Bohren'],
            'Drop_Drehen': per_class_drops['Drehen'],
            'Drop_Fraesen': per_class_drops['Fräsen'],
            'Mean_Gate_Value': mean_gate
        })
    
    # Add ONLY_KEY_PMI analysis if available
    if 'ONLY_KEY_PMI' in results:
        key_results = results['ONLY_KEY_PMI']
        key_f1_per_fold = np.array([r['f1_macro'] for r in key_results])
        
        # Compare with GEOMETRY baseline
        key_vs_geom = key_f1_per_fold - geom_f1_per_fold
        key_gain_mean, key_gain_ci_low, key_gain_ci_high = calculate_bootstrap_ci(key_vs_geom)
        
        # Wilcoxon test vs geometry
        if len(key_vs_geom) >= 5:
            stat_wilcox, p_wilcox = wilcoxon(key_vs_geom, alternative='greater')
        else:
            stat_wilcox, p_wilcox = np.nan, np.nan
        
        # Efficiency calculation
        efficiency = (key_gain_mean / total_pmi_gain) * 100 if total_pmi_gain > 0 else 0
        
        analysis_df.append({
            'Group': 'KEY_PMI (Dim+GeoTol)',
            'Type': 'ONLY',
            'F1_mean': key_f1_per_fold.mean(),
            'F1_std': key_f1_per_fold.std(),
            'Performance_Drop': -(key_gain_mean),  # Negative because it's a gain
            'Drop_CI_Low': -(key_gain_ci_high),  # Inverted for gain
            'Drop_CI_High': -(key_gain_ci_low),
            'Relative_Importance_%': efficiency,
            'Wilcoxon_statistic': stat_wilcox,
            'Wilcoxon_p': p_wilcox,
            'Significant': p_wilcox < ALPHA if not np.isnan(p_wilcox) else False,
            'Drop_Bohren': np.nan,  # Not applicable for ONLY
            'Drop_Drehen': np.nan,
            'Drop_Fraesen': np.nan,
            'Mean_Gate_Value': np.nanmean([r.get('gate_value', np.nan) for r in key_results])
        })
    
    df = pd.DataFrame(analysis_df)
    
    # Separate WITHOUT and ONLY analyses
    df_without = df[df['Type'] == 'WITHOUT'].sort_values('Performance_Drop', ascending=False)
    
    # Apply Holm-Bonferroni multiple testing correction only to WITHOUT comparisons
    if len(df_without) > 0:
        adj_pvals, reject = holm_adjust(df_without['Wilcoxon_p'].values)
        df_without['p_holm'] = adj_pvals
        df_without['Significant'] = reject
    
    # Combine back
    df_final = pd.concat([df_without, df[df['Type'] == 'ONLY']], ignore_index=True)
    
    # Save analysis with timestamp
    df_final.to_csv(OUTPUT_DIR / 'ablation_analysis.csv', index=False)
    logger.info(f"\nAnalysis saved to {OUTPUT_DIR / 'ablation_analysis.csv'}")
    
    # Also save a summary report
    with open(OUTPUT_DIR / 'summary_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("ABLATION STUDY SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("BASELINE PERFORMANCE:\n")
        f.write(f"  GEOMETRY_ONLY: {geom_f1_per_fold.mean():.4f} ± {geom_f1_per_fold.std():.4f}\n")
        f.write(f"  FULL PMI:      {full_f1_per_fold.mean():.4f} ± {full_f1_per_fold.std():.4f}\n")
        f.write(f"  PMI Gain:      +{total_pmi_gain:.4f}\n\n")
        
        f.write("FEATURE GROUP IMPORTANCE:\n")
        for idx, row in df_without.iterrows():
            sig = "***" if row.get('p_holm', 1.0) < 0.001 else ("**" if row.get('p_holm', 1.0) < 0.01 else ("*" if row.get('Significant', False) else ""))
            f.write(f"  {row['Group']:20s}: Drop={row['Performance_Drop']:.4f}, p={row.get('p_holm', row['Wilcoxon_p']):.4f} {sig}\n")
        
        if 'ONLY_KEY_PMI' in results:
            key_row = df_final[df_final['Group'] == 'KEY_PMI (Dim+GeoTol)'].iloc[0]
            f.write(f"\nKEY FEATURES ANALYSIS:\n")
            f.write(f"  F1-Score:     {key_row['F1_mean']:.4f} ± {key_row['F1_std']:.4f}\n")
            f.write(f"  Efficiency:   {key_row['Relative_Importance_%']:.1f}% of total PMI benefit\n")
            f.write(f"  Features:     13/30 (43% of features)\n")
    
    logger.info(f"Summary report saved to {OUTPUT_DIR / 'summary_report.txt'}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📊 FEATURE GROUP IMPORTANCE RANKING (Holm-corrected)")
    logger.info("="*60)
    logger.info("Group                     | Drop    | 95% CI              | p-raw  | p-holm | Sig | Rel.Imp")
    logger.info("-"*60)
    for idx, row in df_without.iterrows():
        sig_marker = "***" if row.get('p_holm', 1.0) < 0.001 else ("**" if row.get('p_holm', 1.0) < 0.01 else ("*" if row.get('Significant', False) else " "))
        logger.info(f"{row['Group']:25s} | {row['Performance_Drop']:6.4f} | [{row['Drop_CI_Low']:5.4f}, {row['Drop_CI_High']:5.4f}] | "
                   f"{row['Wilcoxon_p']:6.4f} | {row.get('p_holm', row['Wilcoxon_p']):6.4f} | {sig_marker:3s} | {row['Relative_Importance_%']:5.1f}%")
    
    if 'ONLY_KEY_PMI' in results:
        logger.info("-"*60)
        key_row = df_final[df_final['Group'] == 'KEY_PMI (Dim+GeoTol)'].iloc[0]
        logger.info(f"{'KEY_PMI (13/30 features)':25s} | {key_row['F1_mean']:6.4f} | "
                   f"Captures {key_row['Relative_Importance_%']:.1f}% of PMI benefit with 43% of features")
    
    logger.info("-"*60)
    logger.info("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    
    return df_final


if __name__ == "__main__":
    try:
        results, analysis = run_feature_removal_ablations()
        logger.info(f"\n" + "="*80)
        logger.info("✓ ABLATION STUDY COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"✓ Results saved in: {OUTPUT_DIR}")
        
        logger.info("\n" + "="*80)
        logger.info("🔑 KEY FINDINGS")
        logger.info("="*80)
        
        # Top WITHOUT groups
        df_without = analysis[analysis['Type'] == 'WITHOUT']
        if len(df_without) > 0:
            top3 = df_without.nlargest(3, 'Performance_Drop')
            logger.info("\n📍 Most important PMI groups (by performance drop):")
            for i, (idx, row) in enumerate(top3.iterrows(), 1):
                sig = "✓ (significant)" if row.get('Significant', False) else "✗ (not significant)"
                logger.info(f"  {i}. {row['Group']}: {row['Relative_Importance_%']:.1f}% of total PMI gain")
                logger.info(f"     Drop: {row['Performance_Drop']:.4f}, p={row['Wilcoxon_p']:.3f} {sig}")
        
        # KEY_PMI efficiency
        if 'KEY_PMI (Dim+GeoTol)' in analysis['Group'].values:
            key_row = analysis[analysis['Group'] == 'KEY_PMI (Dim+GeoTol)'].iloc[0]
            logger.info(f"\n🎯 KEY FEATURES EFFICIENCY:")
            logger.info(f"  • Uses only 13/30 features (43%)")
            logger.info(f"  • Achieves F1={key_row['F1_mean']:.4f}")
            logger.info(f"  • Captures {key_row['Relative_Importance_%']:.1f}% of total PMI benefit")
        
        # Statistical summary
        significant_groups = df_without[df_without.get('Significant', False) == True] if 'Significant' in df_without.columns else pd.DataFrame()
        logger.info(f"\n📈 Statistical summary:")
        logger.info(f"  • Significant groups: {len(significant_groups)}/{len(df_without)}")
        if len(significant_groups) > 0:
            logger.info(f"  • Names: {', '.join(significant_groups['Group'].tolist())}")
            
    except Exception as e:
        logger.error(f"Error during ablation study: {str(e)}", exc_info=True)
        raise