#!/usr/bin/env python3
"""
Cross-Validation for Unified Process Classifier
Geometry-only vs. Multi-modal (Geometry+PMI)
With repeated CV, bootstrap CIs, and permutation tests
"""

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    f1_score, accuracy_score, hamming_loss, multilabel_confusion_matrix, 
    jaccard_score, precision_recall_fscore_support
)
from torch.utils.data import DataLoader, Subset, ConcatDataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import logging
import warnings
warnings.filterwarnings('ignore')

# PyTorch Lightning
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Multilabel stratification
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    USE_STRATIFIED = True
except ImportError:
    print("Warning: iterstrat not installed. Falling back to regular KFold.")
    print("Install with: pip install iterative-stratification")
    from sklearn.model_selection import KFold
    USE_STRATIFIED = False

# Custom imports - Updated for unified model
from mpp.ml.models.classifier.unified_process_classifier import UnifiedProcessClassifier
from mpp.ml.datasets.tkms import TKMS_Process_Dataset
from mpp.ml.datasets.tkms_pmi import TKMS_PMI_Dataset

# ========== CONFIGURATION ==========
N_FOLDS = 5
N_REPEATS = 5  # 5×5 = 25 data points for better statistical power
SEED = 42
BATCH_SIZE = 85
MAX_EPOCHS = 100
PATIENCE = 20
CLASS_NAMES = ["Bohren", "Drehen", "Fräsen"]
NUM_WORKERS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Statistical Test Parameters
N_BOOTSTRAP = 5000  # Reduced from 10000 for faster computation
N_PERMUTATIONS = 5000  # Reduced from 10000 for faster computation
ALPHA = 0.05  # Significance level
USE_FDR = False  # Set to True only if reviewers request it

# Output Directory
OUTPUT_DIR = Path("cv_results_unified") / datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(OUTPUT_DIR / 'cv_log.txt')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Best hyperparameters from tuning
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
    "modality_dropout": 0.0
}

HP_PMI = {
    "dropout": 0.280,
    "lr": 0.000690,
    "embed_dim": 64,
    "num_layers": 3,
    "num_heads": 8,
    "weight_decay": 0.000277,
    "use_pmi": True,  # Multi-modal
    "pmi_dim": 30,
    "initial_gate": 0.171,
    "modality_dropout": 0.206
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
    logger.info(f"Set seed to {seed}")


def load_datasets_with_alignment():
    """Load and align datasets ensuring same order"""
    logger.info("Loading and aligning datasets...")
    
    train_geom = TKMS_Process_Dataset(mode="train", target_type="step-set")
    valid_geom = TKMS_Process_Dataset(mode="valid", target_type="step-set")
    train_pmi = TKMS_PMI_Dataset(mode="train", target_type="step-set", **PMI_CONFIG)
    valid_pmi = TKMS_PMI_Dataset(mode="valid", target_type="step-set", **PMI_CONFIG)
    
    assert train_geom.samples == train_pmi.samples, "Train samples mismatch!"
    assert valid_geom.samples == valid_pmi.samples, "Valid samples mismatch!"
    
    all_sample_ids = train_geom.samples + valid_geom.samples
    
    dataset_geom = ConcatDataset([train_geom, valid_geom])
    dataset_pmi = ConcatDataset([train_pmi, valid_pmi])
    
    all_labels = []
    for i in range(len(train_geom) + len(valid_geom)):
        if i < len(train_geom):
            _, label = train_geom[i]
        else:
            _, label = valid_geom[i - len(train_geom)]
        all_labels.append(label.numpy())
    
    labels = np.array(all_labels)
    
    # Add interaction columns for better stratification
    # Original: B, D, F (Bohren, Drehen, Fräsen)
    # Add: B∧D, B∧F, D∧F
    # Convert to int for bitwise operations
    labels_int = labels.astype(int)
    interactions = np.zeros((len(labels), 3), dtype=int)
    interactions[:, 0] = labels_int[:, 0] & labels_int[:, 1]  # B∧D
    interactions[:, 1] = labels_int[:, 0] & labels_int[:, 2]  # B∧F
    interactions[:, 2] = labels_int[:, 1] & labels_int[:, 2]  # D∧F
    
    # Augmented labels for stratification
    labels_augmented = np.hstack([labels_int, interactions])
    
    assert len(dataset_geom) == len(dataset_pmi) == len(labels), "Dataset size mismatch!"
    
    logger.info(f"Total samples: {len(labels)}")
    logger.info(f"Label distribution: {labels.sum(axis=0)} ({CLASS_NAMES})")
    logger.info(f"Interaction distribution: B∧D={interactions[:, 0].sum()}, "
                f"B∧F={interactions[:, 1].sum()}, D∧F={interactions[:, 2].sum()}")
    
    return dataset_geom, dataset_pmi, labels, labels_augmented, all_sample_ids


def calculate_bootstrap_ci(values1, values2, n_bootstrap=N_BOOTSTRAP, seed=42):
    """
    Calculate bootstrap confidence interval for paired differences
    
    Parameters:
    -----------
    values1, values2: arrays of paired values
    n_bootstrap: number of bootstrap samples
    
    Returns:
    --------
    mean_diff, ci_low, ci_high, p_value
    """
    diff = np.array(values1) - np.array(values2)
    rng = np.random.default_rng(seed)
    
    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(diff), len(diff))
        boot_means.append(diff[idx].mean())
    
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    mean_diff = diff.mean()
    
    # Bootstrap p-value (proportion of bootstrap samples < 0)
    p_value = (np.sum(np.array(boot_means) <= 0) + 1) / (n_bootstrap + 1)
    
    return mean_diff, ci_low, ci_high, p_value


def permutation_test(values1, values2, n_permutations=N_PERMUTATIONS, seed=42):
    """
    Paired permutation test
    
    Returns:
    --------
    p_value (one-sided, testing if values1 > values2)
    """
    diff = np.array(values1) - np.array(values2)
    obs_mean = diff.mean()
    
    rng = np.random.default_rng(seed)
    perm_means = []
    
    for _ in range(n_permutations):
        signs = np.where(rng.random(len(diff)) < 0.5, 1, -1)
        perm_means.append((diff * signs).mean())
    
    # One-sided p-value
    p_value = (np.sum(np.array(perm_means) >= obs_mean) + 1) / (n_permutations + 1)
    return p_value


def find_optimal_thresholds_on_train(model, train_loader, train_indices, dataset, labels_augmented, use_pmi=False, per_class=True, val_split=0.1, fold_seed=42):
    """Find optimal thresholds on inner train/val split to avoid leakage
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    train_loader : DataLoader
        Training data loader
    train_indices : list
        Indices for training fold
    dataset : Dataset
        Full dataset
    labels_augmented : numpy.ndarray
        Augmented labels including interactions for stratification
    use_pmi : bool
        Whether using PMI features
    per_class : bool
        Whether to find per-class thresholds
    val_split : float
        Proportion for inner validation
    fold_seed : int
        Seed for reproducibility (should be fold-specific)
    """
    # Get augmented labels for stratified split
    y_train_augmented = labels_augmented[train_indices]
    
    # Create stratified inner split
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=fold_seed)
        inner_train_rel, inner_val_rel = next(msss.split(np.zeros(len(y_train_augmented)), y_train_augmented))
    except ImportError:
        # Fallback to random split if iterstrat not available
        logger.warning("MultilabelStratifiedShuffleSplit not available, using random split")
        rng = np.random.default_rng(fold_seed)
        n_train = len(train_indices)
        n_inner_val = int(n_train * val_split)
        inner_indices = rng.permutation(n_train)
        inner_train_rel = inner_indices[n_inner_val:]
        inner_val_rel = inner_indices[:n_inner_val]
    
    # Map back to original indices
    inner_train_indices = [train_indices[i] for i in inner_train_rel]
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
        thresholds = np.zeros(len(CLASS_NAMES))
        for class_idx in range(len(CLASS_NAMES)):
            # Use sklearn's precision_recall_curve for more efficient threshold search
            from sklearn.metrics import precision_recall_curve
            precision, recall, thresholds_pr = precision_recall_curve(
                all_labels[:, class_idx], 
                all_probs[:, class_idx]
            )
            # F1 = 2 * (precision * recall) / (precision + recall)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            if best_idx < len(thresholds_pr):
                thresholds[class_idx] = thresholds_pr[best_idx]
            else:
                thresholds[class_idx] = 0.5
    else:
        best_threshold = 0.5
        best_f1 = 0.0
        for threshold in np.arange(0.1, 0.9, 0.01):
            preds = (all_probs > threshold).astype(int)
            f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds = best_threshold
    
    logger.debug(f"Optimal thresholds found on inner val: {thresholds}")
    logger.debug(f"Inner split sizes - train: {len(inner_train_indices)}, val: {len(inner_val_indices)}")
    
    return thresholds


def evaluate_model(model, loader, thresholds, use_pmi=False):
    """Evaluate model with comprehensive metrics"""
    all_probs = []
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if use_pmi:
                (inputs, pmi), labels = batch
                outputs = model(inputs.to(DEVICE), pmi.to(DEVICE))
            else:
                inputs, labels = batch
                outputs = model(inputs.to(DEVICE))
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            if isinstance(thresholds, np.ndarray):
                preds = np.zeros_like(probs, dtype=int)
                for i in range(len(CLASS_NAMES)):
                    preds[:, i] = (probs[:, i] > thresholds[i]).astype(int)
            else:
                preds = (probs > thresholds).astype(int)
            
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    metrics = {
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'subset_accuracy': accuracy_score(all_labels, all_preds),
        'jaccard_samples': jaccard_score(all_labels, all_preds, average='samples', zero_division=0),
        'jaccard_macro': jaccard_score(all_labels, all_preds, average='macro', zero_division=0),
        'hamming_loss': hamming_loss(all_labels, all_preds),
    }
    
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    for i, class_name in enumerate(CLASS_NAMES):
        metrics[f'f1_{class_name}'] = f1[i]
        metrics[f'precision_{class_name}'] = precision[i]
        metrics[f'recall_{class_name}'] = recall[i]
        metrics[f'support_{class_name}'] = support[i]
    
    cm_per_class = multilabel_confusion_matrix(all_labels, all_preds)
    
    return metrics, all_preds, all_labels, all_probs, cm_per_class


def train_fold(repeat_idx, fold_idx, train_idx, val_idx, dataset, model_type, sample_ids, labels_augmented):
    """Train one fold with tracking"""
    global_fold_idx = repeat_idx * N_FOLDS + fold_idx
    fold_seed = SEED + global_fold_idx
    ensure_determinism(fold_seed)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_type.upper()} - Repeat {repeat_idx+1}/{N_REPEATS}, Fold {fold_idx+1}/{N_FOLDS}")
    logger.info(f"Global Fold: {global_fold_idx+1}/{N_REPEATS*N_FOLDS}")
    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Use unified model for both modes
    model_class = UnifiedProcessClassifier
    
    if model_type == 'pmi':
        hp = HP_PMI.copy()
        use_pmi = True
    else:
        hp = HP_GEOM.copy()
        use_pmi = False
    
    hp['max_epochs'] = MAX_EPOCHS
    
    PERSIST = NUM_WORKERS > 0
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=PERSIST, prefetch_factor=2 if PERSIST else None
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=PERSIST, prefetch_factor=2 if PERSIST else None
    )
    
    model = model_class(**hp)
    
    checkpoint_dir = OUTPUT_DIR / f"repeat_{repeat_idx}" / f"fold_{fold_idx}" / model_type
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
        mode='min',
        verbose=False
    )
    
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stop],
        enable_progress_bar=False,
        logger=False,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        gradient_clip_val=1.0,
        deterministic=True
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    model = model_class.load_from_checkpoint(checkpoint_callback.best_model_path)
    model.to(DEVICE)
    model.eval()
    
    # Pass dataset and labels_augmented to threshold finding function
    thresholds = find_optimal_thresholds_on_train(
        model, train_loader, train_idx, dataset, labels_augmented, 
        use_pmi, per_class=True, fold_seed=fold_seed
    )
    metrics, preds, labels, probs, cm = evaluate_model(model, val_loader, thresholds, use_pmi)
    
    metrics['repeat'] = repeat_idx
    metrics['fold'] = fold_idx
    metrics['global_fold'] = global_fold_idx
    metrics['epochs_trained'] = trainer.current_epoch + 1
    metrics['model_type'] = model_type
    
    # Add gate value for PMI model
    if use_pmi and hasattr(model, 'gate'):
        metrics['gate_value'] = torch.sigmoid(model.gate).item()
        logger.info(f"  Gate value: {metrics['gate_value']:.3f}")
    
    logger.info(f"  F1-Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"  Epochs: {metrics['epochs_trained']}")
    
    return metrics


def perform_statistical_tests(results_geom, results_pmi):
    """
    Perform comprehensive statistical tests with FDR correction
    """
    test_results = {}
    metrics_to_test = ['f1_macro', 'f1_micro', 'jaccard_samples', 'subset_accuracy']
    
    # Add per-class F1 scores
    for class_name in CLASS_NAMES:
        metrics_to_test.append(f'f1_{class_name}')
    
    all_p_values = []
    all_test_names = []
    
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL TESTS")
    logger.info("="*80)
    
    for metric in metrics_to_test:
        geom_values = [r[metric] for r in results_geom]
        pmi_values = [r[metric] for r in results_pmi]
        
        # 1. Wilcoxon signed-rank test (one-sided)
        if len(geom_values) >= 5:
            stat_wilcox, p_wilcox = wilcoxon(pmi_values, geom_values, alternative='greater')
        else:
            stat_wilcox, p_wilcox = np.nan, np.nan
        
        # 2. Bootstrap CI
        mean_diff, ci_low, ci_high, p_bootstrap = calculate_bootstrap_ci(
            pmi_values, geom_values, N_BOOTSTRAP
        )
        
        # 3. Permutation test
        p_perm = permutation_test(pmi_values, geom_values, N_PERMUTATIONS)
        
        test_results[metric] = {
            'mean_geom': np.mean(geom_values),
            'std_geom': np.std(geom_values),
            'mean_pmi': np.mean(pmi_values),
            'std_pmi': np.std(pmi_values),
            'mean_diff': mean_diff,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'relative_improvement': (mean_diff / np.mean(geom_values)) * 100 if np.mean(geom_values) > 0 else 0,
            'wilcoxon_statistic': stat_wilcox,
            'wilcoxon_p': p_wilcox,
            'bootstrap_p': p_bootstrap,
            'permutation_p': p_perm
        }
        
        # Collect p-values for FDR correction
        if not np.isnan(p_wilcox):
            all_p_values.append(p_wilcox)
            all_test_names.append(metric)
    
    # FDR correction
    if USE_FDR and len(all_p_values) > 0:
        rejected, p_adjusted, _, _ = multipletests(all_p_values, alpha=ALPHA, method='fdr_bh')
        for i, metric in enumerate(all_test_names):
            test_results[metric]['wilcoxon_p_adjusted'] = p_adjusted[i]
            test_results[metric]['significant_fdr'] = rejected[i]
    
    # Print results
    logger.info(f"\nResults based on {len(results_geom)} samples (N_REPEATS={N_REPEATS}, N_FOLDS={N_FOLDS})")
    logger.info("-" * 80)
    
    for metric in metrics_to_test:
        res = test_results[metric]
        logger.info(f"\n{metric}:")
        logger.info(f"  Geometry:     {res['mean_geom']:.4f} ± {res['std_geom']:.4f}")
        logger.info(f"  Geometry+PMI: {res['mean_pmi']:.4f} ± {res['std_pmi']:.4f}")
        logger.info(f"  Difference:   {res['mean_diff']:.4f} ({res['relative_improvement']:+.1f}%)")
        logger.info(f"  95% CI:       [{res['ci_low']:.4f}, {res['ci_high']:.4f}]")
        
        if not np.isnan(res['wilcoxon_p']):
            sig_marker = "✓" if res['wilcoxon_p'] < ALPHA else "✗"
            logger.info(f"  Wilcoxon p:   {res['wilcoxon_p']:.4f} {sig_marker}")
            
            if USE_FDR and 'wilcoxon_p_adjusted' in res:
                sig_marker_fdr = "✓" if res['significant_fdr'] else "✗"
                logger.info(f"  FDR-adjusted: {res['wilcoxon_p_adjusted']:.4f} {sig_marker_fdr}")
        
        logger.info(f"  Bootstrap p:  {res['bootstrap_p']:.4f}")
        logger.info(f"  Permutation p: {res['permutation_p']:.4f}")
    
    return test_results


def create_visualizations(results_geom, results_pmi, test_results, output_dir):
    """Create visualization plots"""
    df_geom = pd.DataFrame(results_geom)
    df_pmi = pd.DataFrame(results_pmi)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    # 1. Box plot with significance markers
    ax = axes[0, 0]
    positions = [1, 2]
    data = [df_geom['f1_macro'], df_pmi['f1_macro']]
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax.set_xticks(positions)
    ax.set_xticklabels(['Geometry', 'Geometry+PMI'])
    ax.set_ylabel('F1-Macro Score')
    ax.set_title('F1-Macro Distribution (N=25)')
    
    # Add significance marker if significant
    if test_results['f1_macro']['wilcoxon_p'] < ALPHA:
        y_max = max(max(df_geom['f1_macro']), max(df_pmi['f1_macro']))
        ax.plot([1, 2], [y_max + 0.01, y_max + 0.01], 'k-')
        ax.text(1.5, y_max + 0.015, '***', ha='center', fontsize=12)
    
    ax.grid(True, alpha=0.3)
    
    # 2. Paired differences with CI
    ax = axes[0, 1]
    metrics = ['f1_macro', 'f1_micro', 'jaccard_samples']
    metric_labels = ['F1-Macro', 'F1-Micro', 'Jaccard']
    
    means = [test_results[m]['mean_diff'] for m in metrics]
    ci_lows = [test_results[m]['ci_low'] for m in metrics]
    ci_highs = [test_results[m]['ci_high'] for m in metrics]
    
    x = np.arange(len(metrics))
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(ci_lows), 
                                 np.array(ci_highs) - np.array(means)],
                fmt='o', capsize=5, capthick=2, markersize=8)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel('Mean Difference (PMI - Geom)')
    ax.set_title('Mean Differences with 95% Bootstrap CI')
    ax.grid(True, alpha=0.3)
    
    # 3. Per-repeat performance
    ax = axes[0, 2]
    for repeat in range(N_REPEATS):
        repeat_geom = [r['f1_macro'] for r in results_geom if r['repeat'] == repeat]
        repeat_pmi = [r['f1_macro'] for r in results_pmi if r['repeat'] == repeat]
        x = np.arange(len(repeat_geom))
        ax.plot(x, repeat_geom, 'o-', alpha=0.5, label=f'Geom R{repeat+1}' if repeat == 0 else '')
        ax.plot(x, repeat_pmi, 's-', alpha=0.5, label=f'PMI R{repeat+1}' if repeat == 0 else '')
    ax.set_xlabel('Fold within Repeat')
    ax.set_ylabel('F1-Macro')
    ax.set_title('Performance Across Repeats')
    ax.legend(['Geometry', 'Geometry+PMI'])
    ax.grid(True, alpha=0.3)
    
    # 4. Distribution of differences
    ax = axes[1, 0]
    differences = np.array([r['f1_macro'] for r in results_pmi]) - \
                  np.array([r['f1_macro'] for r in results_geom])
    ax.hist(differences, bins=15, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(0, color='red', linestyle='--', label='No difference')
    ax.axvline(np.mean(differences), color='blue', linestyle='-', linewidth=2, label='Mean difference')
    ax.set_xlabel('F1-Macro Difference (PMI - Geom)')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Paired Differences (n={len(differences)})')
    ax.legend()
    
    # 5. Per-class performance
    ax = axes[1, 1]
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    
    improvements = [test_results[f'f1_{c}']['relative_improvement'] for c in CLASS_NAMES]
    p_values = [test_results[f'f1_{c}']['wilcoxon_p'] for c in CLASS_NAMES]
    
    bars = ax.bar(x, improvements, width, alpha=0.8)
    
    # Color bars based on significance
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        if p < 0.001:
            bar.set_color('darkgreen')
        elif p < 0.01:
            bar.set_color('green')
        elif p < 0.05:
            bar.set_color('lightgreen')
        else:
            bar.set_color('gray')
    
    ax.set_ylabel('Relative Improvement (%)')
    ax.set_xlabel('Process Class')
    ax.set_title('Per-Class Relative Improvement')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # 6. Learning efficiency and gate values
    ax = axes[1, 2]
    if 'gate_value' in df_pmi.columns:
        ax2 = ax.twinx()
        epochs_geom = [r['epochs_trained'] for r in results_geom]
        epochs_pmi = [r['epochs_trained'] for r in results_pmi]
        gate_values = df_pmi['gate_value'].values
        
        bp1 = ax.boxplot([epochs_geom, epochs_pmi], labels=['Geometry', 'Geometry+PMI'],
                         positions=[1, 2], widths=0.4, patch_artist=True)
        bp1['boxes'][0].set_facecolor('lightblue')
        bp1['boxes'][1].set_facecolor('lightgreen')
        
        ax.set_ylabel('Epochs until convergence', color='black')
        ax.tick_params(axis='y', labelcolor='black')
        
        ax2.scatter(np.ones(len(gate_values)) * 2.4, gate_values, alpha=0.6, color='red', s=30)
        ax2.set_ylabel('Gate Value', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 1)
        
        ax.set_title('Training Efficiency & Gate Values')
    else:
        epochs_geom = [r['epochs_trained'] for r in results_geom]
        epochs_pmi = [r['epochs_trained'] for r in results_pmi]
        ax.boxplot([epochs_geom, epochs_pmi], labels=['Geometry', 'Geometry+PMI'])
        ax.set_ylabel('Epochs until convergence')
        ax.set_title('Training Efficiency')
    
    ax.grid(True, alpha=0.3)
    
    # 7. Correlation between metric improvements
    ax = axes[2, 0]
    f1_diff = np.array([r['f1_macro'] for r in results_pmi]) - \
              np.array([r['f1_macro'] for r in results_geom])
    jaccard_diff = np.array([r['jaccard_samples'] for r in results_pmi]) - \
                   np.array([r['jaccard_samples'] for r in results_geom])
    
    ax.scatter(f1_diff, jaccard_diff, alpha=0.6)
    ax.set_xlabel('F1-Macro Improvement')
    ax.set_ylabel('Jaccard Improvement')
    ax.set_title('Correlation of Metric Improvements')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # 8. Statistical summary table
    ax = axes[2, 1]
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = []
    for metric in ['f1_macro', 'f1_micro', 'jaccard_samples']:
        res = test_results[metric]
        summary_data.append([
            metric.replace('_', '-'),
            f"{res['mean_diff']:.4f}",
            f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}]",
            f"{res['wilcoxon_p']:.4f}",
            "✓" if res['wilcoxon_p'] < ALPHA else "✗"
        ])
    
    table = ax.table(cellText=summary_data,
                     colLabels=['Metric', 'Mean Δ', '95% CI', 'p-value', 'Sig.'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Statistical Summary', fontsize=12, pad=20)
    
    # 9. Relative improvement heatmap
    ax = axes[2, 2]
    improvements_matrix = pd.DataFrame({
        'F1-Macro': [test_results['f1_macro']['relative_improvement']],
        'F1-Micro': [test_results['f1_micro']['relative_improvement']],
        'Jaccard': [test_results['jaccard_samples']['relative_improvement']],
        'Subset Acc': [test_results['subset_accuracy']['relative_improvement']]
    })
    
    sns.heatmap(improvements_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Improvement %'}, ax=ax, vmin=-5, vmax=20)
    ax.set_title('Relative Improvements Summary')
    ax.set_yticklabels(['PMI vs Geom'], rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cv_results_unified.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plots to {output_dir / 'cv_results_unified.png'}")


def run_repeated_cross_validation():
    """Main repeated cross-validation function"""
    logger.info("="*80)
    logger.info("CROSS-VALIDATION: Unified Process Classifier")
    logger.info("Geometry-only vs. Multi-modal (Geometry+PMI)")
    logger.info(f"Configuration: {N_REPEATS}×{N_FOLDS} = {N_REPEATS*N_FOLDS} total folds")
    logger.info(f"Statistical tests: Wilcoxon (one-sided), Bootstrap CI, Permutation test")
    logger.info(f"FDR correction: {'Yes' if USE_FDR else 'No'}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("="*80)
    
    # Save configuration
    config = {
        'n_folds': N_FOLDS,
        'n_repeats': N_REPEATS,
        'total_folds': N_FOLDS * N_REPEATS,
        'seed': SEED,
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'patience': PATIENCE,
        'n_bootstrap': N_BOOTSTRAP,
        'n_permutations': N_PERMUTATIONS,
        'alpha': ALPHA,
        'use_fdr': USE_FDR,
        'model_class': 'UnifiedProcessClassifier',
        'hp_geom': HP_GEOM,
        'hp_pmi': HP_PMI,
        'pmi_config': PMI_CONFIG,
        'class_names': CLASS_NAMES
    }
    
    with open(OUTPUT_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Load data
    dataset_geom, dataset_pmi, labels, labels_augmented, sample_ids = load_datasets_with_alignment()
    
    # Store all results
    results_geom = []
    results_pmi = []
    
    # Repeated Cross-Validation
    for repeat_idx in range(N_REPEATS):
        logger.info(f"\n{'='*80}")
        logger.info(f"REPEAT {repeat_idx + 1}/{N_REPEATS}")
        logger.info("="*80)
        
        # New seed for each repeat
        repeat_seed = SEED + repeat_idx * 1000
        
        if USE_STRATIFIED:
            splitter = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=repeat_seed)
            split_iter = splitter.split(np.zeros(len(labels_augmented)), labels_augmented)
        else:
            splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=repeat_seed)
            split_iter = splitter.split(labels)
        
        for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
            # Train both models on same fold
            for model_type, dataset in [('geom', dataset_geom), ('pmi', dataset_pmi)]:
                metrics = train_fold(
                    repeat_idx, fold_idx, train_idx, val_idx, 
                    dataset, model_type, sample_ids, labels_augmented
                )
                
                if model_type == 'geom':
                    results_geom.append(metrics)
                else:
                    results_pmi.append(metrics)
    
    # Perform statistical tests
    test_results = perform_statistical_tests(results_geom, results_pmi)
    
    # Create visualizations
    create_visualizations(results_geom, results_pmi, test_results, OUTPUT_DIR)
    
    # Save all results
    final_results = {
        'config': config,
        'results_geom': results_geom,
        'results_pmi': results_pmi,
        'statistical_tests': test_results,
        'summary': {
            'n_total_folds': len(results_geom),
            'main_result': {
                'metric': 'f1_macro',
                'geom_mean': test_results['f1_macro']['mean_geom'],
                'pmi_mean': test_results['f1_macro']['mean_pmi'],
                'improvement': test_results['f1_macro']['relative_improvement'],
                'p_value': test_results['f1_macro']['wilcoxon_p'],
                'significant': test_results['f1_macro']['wilcoxon_p'] < ALPHA
            }
        }
    }
    
    with open(OUTPUT_DIR / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=4, default=float)
    
    # Save DataFrames
    pd.DataFrame(results_geom).to_csv(OUTPUT_DIR / 'results_geom.csv', index=False)
    pd.DataFrame(results_pmi).to_csv(OUTPUT_DIR / 'results_pmi.csv', index=False)
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    
    main_res = test_results['f1_macro']
    logger.info(f"\nMain Result (F1-Macro):")
    logger.info(f"  Geometry:     {main_res['mean_geom']:.4f} ± {main_res['std_geom']:.4f}")
    logger.info(f"  Geometry+PMI: {main_res['mean_pmi']:.4f} ± {main_res['std_pmi']:.4f}")
    logger.info(f"  Improvement:  {main_res['relative_improvement']:+.1f}%")
    logger.info(f"  95% CI:       [{main_res['ci_low']:.4f}, {main_res['ci_high']:.4f}]")
    logger.info(f"  p-value:      {main_res['wilcoxon_p']:.4f}")
    
    if main_res['wilcoxon_p'] < ALPHA:
        logger.info(f"  → ✓ SIGNIFICANT at α={ALPHA}")
    else:
        logger.info(f"  → ✗ Not significant at α={ALPHA}")
    
    # Print gate value statistics for PMI models
    if any('gate_value' in r for r in results_pmi):
        gate_values = [r['gate_value'] for r in results_pmi if 'gate_value' in r]
        logger.info(f"\nGate value statistics:")
        logger.info(f"  Mean: {np.mean(gate_values):.3f}")
        logger.info(f"  Std:  {np.std(gate_values):.3f}")
        logger.info(f"  Range: [{np.min(gate_values):.3f}, {np.max(gate_values):.3f}]")
    
    return final_results


if __name__ == "__main__":
    try:
        results = run_repeated_cross_validation()
        logger.info(f"\n✓ Cross-Validation completed successfully!")
        logger.info(f"✓ Results saved in: {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"Error during cross-validation: {str(e)}", exc_info=True)
        raise