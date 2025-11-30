#!/usr/bin/env python3
"""
Cross-Validation for Unified Process Classifier
Extended with detailed confusion matrix analysis, error patterns, and confidence tracking
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

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    USE_STRATIFIED = True
except ImportError:
    from sklearn.model_selection import KFold
    USE_STRATIFIED = False

from mpp.ml.models.classifier.unified_process_classifier import UnifiedProcessClassifier
from mpp.ml.datasets.tkms import TKMS_Process_Dataset
from mpp.ml.datasets.tkms_pmi import TKMS_PMI_Dataset

# Configuration
N_FOLDS = 5
N_REPEATS = 5
SEED = 42
BATCH_SIZE = 85
MAX_EPOCHS = 100
PATIENCE = 20
CLASS_NAMES = ["Bohren", "Drehen", "Fräsen"]
NUM_WORKERS = 2  # Reduced from 3 to avoid worker crashes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BOOTSTRAP = 5000
N_PERMUTATIONS = 5000
ALPHA = 0.05
USE_FDR = False

OUTPUT_DIR = Path("cv_results_extended") / datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(OUTPUT_DIR / 'cv_log.txt')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Best hyperparameters from tuning (2025-11-26)
HP_GEOM = {
    # Geometry Encoder
    "embed_dim": 128,
    "num_layers": 5,
    "num_heads": 16,
    "dropout": 0.154,
    # Optimizer
    "lr": 0.000165,
    "weight_decay": 4.19e-05,
    # PMI settings (not used but required)
    "use_pmi": False,
    "pmi_dim": 30,
    "initial_gate": 0.2,
    "modality_dropout": 0.0,
    # PMI Encoder (not used but required for model init)
    "pmi_hidden_dim": 128,
    "pmi_num_layers": 2,
    "pmi_dropout": 0.2,
    # Fusion (not used but required for model init)
    "fusion_hidden_dim": 128,
    "fusion_num_layers": 1,
    "fusion_dropout": 0.2,
}

HP_PMI = {
    # Geometry Encoder
    "embed_dim": 128,
    "num_layers": 5,
    "num_heads": 16,
    "dropout": 0.344,
    # Optimizer
    "lr": 0.000154,
    "weight_decay": 0.000164,
    # PMI settings
    "use_pmi": True,
    "pmi_dim": 30,
    "initial_gate": 0.412,
    "modality_dropout": 0.0,
    # PMI Encoder (NEW - from tuning)
    "pmi_hidden_dim": 256,
    "pmi_num_layers": 1,
    "pmi_dropout": 0.339,
    # Fusion (NEW - from tuning)
    "fusion_hidden_dim": 128,
    "fusion_num_layers": 1,
    "fusion_dropout": 0.417,
}

PMI_CONFIG = {
    "pmi_path": "/workspace/masterthesis_cadtoplan_fabian_heinze/mpp/encoding_results/standard_encoding.npy",
    "pmi_csv_path": "/workspace/masterthesis_cadtoplan_fabian_heinze/mpp/data_pmi/pmi_features.csv",
    "clip_value": 5.0
}


def ensure_determinism(seed):
    """Ensure reproducible results"""
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def categorize_error_type(true_classes, pred_classes):
    """Categorize error type for multi-label classification"""
    true_set = set(true_classes)
    pred_set = set(pred_classes)
    
    if len(pred_set) == 0 and len(true_set) > 0:
        return 'complete_miss'
    if len(true_set) == 0 and len(pred_set) > 0:
        return 'false_alarm'
    if len(true_set) > 0 and len(pred_set) > 0 and len(true_set & pred_set) == 0:
        return 'complete_substitution'
    if len(true_set & pred_set) > 0:
        if len(pred_set) > len(true_set):
            return 'over_prediction'
        elif len(pred_set) < len(true_set):
            return 'under_prediction'
        else:
            return 'partial_substitution'
    return 'other'


def analyze_error_patterns(labels, preds):
    """Analyze error patterns across all samples"""
    error_types = {
        'complete_miss': 0,
        'false_alarm': 0,
        'complete_substitution': 0,
        'over_prediction': 0,
        'under_prediction': 0,
        'partial_substitution': 0,
        'other': 0
    }
    
    specific_patterns = {}
    
    for i in range(len(labels)):
        if not np.array_equal(labels[i], preds[i]):
            true_classes = [CLASS_NAMES[j] for j in range(len(CLASS_NAMES)) if labels[i][j] == 1]
            pred_classes = [CLASS_NAMES[j] for j in range(len(CLASS_NAMES)) if preds[i][j] == 1]
            
            # Categorize error type
            error_type = categorize_error_type(true_classes, pred_classes)
            error_types[error_type] += 1
            
            # Track specific pattern
            true_str = ', '.join(true_classes) if true_classes else 'None'
            pred_str = ', '.join(pred_classes) if pred_classes else 'None'
            pattern = f"[{true_str}] → [{pred_str}]"
            specific_patterns[pattern] = specific_patterns.get(pattern, 0) + 1
    
    return {
        'error_types': error_types,
        'specific_patterns': specific_patterns,
        'total_errors': sum(error_types.values())
    }


def extract_confidences_per_class(labels, preds, probs):
    """Extract confidence scores for FN and FP per class"""
    confidences = {}
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        # False Negatives: true=1, pred=0
        fn_mask = (labels[:, class_idx] == 1) & (preds[:, class_idx] == 0)
        fn_confidences = probs[fn_mask, class_idx].tolist() if fn_mask.any() else []
        
        # False Positives: true=0, pred=1
        fp_mask = (labels[:, class_idx] == 0) & (preds[:, class_idx] == 1)
        fp_confidences = probs[fp_mask, class_idx].tolist() if fp_mask.any() else []
        
        confidences[f'{class_name}_fn'] = fn_confidences
        confidences[f'{class_name}_fp'] = fp_confidences
    
    return confidences


def load_datasets_with_alignment():
    """Load and align datasets ensuring same order - only train+valid for CV"""
    logger.info("Loading and aligning datasets...")
    
    train_geom = TKMS_Process_Dataset(mode="train", target_type="step-set")
    valid_geom = TKMS_Process_Dataset(mode="valid", target_type="step-set")
    train_pmi = TKMS_PMI_Dataset(
        mode="train",
        target_type="step-set",
        pmi_path=PMI_CONFIG["pmi_path"],
        pmi_csv_path=PMI_CONFIG["pmi_csv_path"],
        clip_value=PMI_CONFIG["clip_value"]
    )
    valid_pmi = TKMS_PMI_Dataset(
        mode="valid",
        target_type="step-set",
        pmi_path=PMI_CONFIG["pmi_path"],
        pmi_csv_path=PMI_CONFIG["pmi_csv_path"],
        clip_value=PMI_CONFIG["clip_value"]
    )
    
    # Verify alignment
    assert train_geom.samples == train_pmi.samples, "Train samples mismatch!"
    assert valid_geom.samples == valid_pmi.samples, "Valid samples mismatch!"
    
    all_sample_ids = train_geom.samples + valid_geom.samples
    
    dataset_geom = ConcatDataset([train_geom, valid_geom])
    dataset_pmi = ConcatDataset([train_pmi, valid_pmi])
    
    # Extract labels
    all_labels = []
    for i in range(len(train_geom) + len(valid_geom)):
        if i < len(train_geom):
            _, label = train_geom[i]
        else:
            _, label = valid_geom[i - len(train_geom)]
        all_labels.append(label.numpy())
    
    labels = np.array(all_labels)
    
    # Create augmented labels with interactions for stratification
    labels_int = labels.astype(int)
    interactions = np.zeros((len(labels), 3), dtype=int)
    interactions[:, 0] = labels_int[:, 0] & labels_int[:, 1]  # Bohren ∧ Drehen
    interactions[:, 1] = labels_int[:, 0] & labels_int[:, 2]  # Bohren ∧ Fräsen
    interactions[:, 2] = labels_int[:, 1] & labels_int[:, 2]  # Drehen ∧ Fräsen
    
    labels_augmented = np.hstack([labels_int, interactions])
    
    assert len(dataset_geom) == len(dataset_pmi) == len(labels), "Dataset size mismatch!"
    
    logger.info(f"Total samples: {len(labels)}")
    logger.info(f"Label distribution: {labels.sum(axis=0)} ({CLASS_NAMES})")
    logger.info(f"Interaction distribution: B∧D={interactions[:, 0].sum()}, "
                f"B∧F={interactions[:, 1].sum()}, D∧F={interactions[:, 2].sum()}")
    
    return dataset_geom, dataset_pmi, labels, labels_augmented, all_sample_ids


def find_optimal_thresholds_on_train(model, train_loader, train_indices, dataset, labels_augmented, use_pmi=False, per_class=True, val_split=0.1, fold_seed=42):
    """Find optimal thresholds on inner train/val split"""
    from sklearn.metrics import precision_recall_curve
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    
    y_train_augmented = labels_augmented[train_indices]
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=fold_seed)
    inner_train_rel, inner_val_rel = next(msss.split(np.zeros(len(y_train_augmented)), y_train_augmented))
    inner_val_indices = [train_indices[i] for i in inner_val_rel]
    
    inner_val_subset = Subset(dataset, inner_val_indices)
    inner_val_loader = DataLoader(
        inner_val_subset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0
    )
    
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
            precision, recall, thresholds_pr = precision_recall_curve(all_labels[:, class_idx], all_probs[:, class_idx])
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
    
    return thresholds


def evaluate_model(model, loader, thresholds, use_pmi=False):
    """Evaluate model and return detailed results"""
    all_preds = []
    all_labels = []
    all_probs = []
    
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
            
            # Apply thresholds
            if isinstance(thresholds, np.ndarray):
                preds = np.zeros_like(probs, dtype=int)
                for i in range(len(CLASS_NAMES)):
                    preds[:, i] = (probs[:, i] > thresholds[i]).astype(int)
            else:
                preds = (probs > thresholds).astype(int)
            
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    # Calculate metrics
    metrics = {
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'accuracy': accuracy_score(all_labels, all_preds),
        'hamming': hamming_loss(all_labels, all_preds),
        'jaccard_samples': jaccard_score(all_labels, all_preds, average='samples', zero_division=0),
        'subset_accuracy': (all_labels == all_preds).all(axis=1).mean()
    }
    
    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    for i, class_name in enumerate(CLASS_NAMES):
        metrics[f'f1_{class_name}'] = f1[i]
        metrics[f'precision_{class_name}'] = precision[i]
        metrics[f'recall_{class_name}'] = recall[i]
    
    # Confusion matrices
    cm_per_class = multilabel_confusion_matrix(all_labels, all_preds)
    
    # Error patterns
    error_analysis = analyze_error_patterns(all_labels, all_preds)
    
    # Confidences
    confidences = extract_confidences_per_class(all_labels, all_preds, all_probs)
    
    return metrics, all_preds, all_labels, all_probs, cm_per_class, error_analysis, confidences


def train_fold(repeat_idx, fold_idx, train_idx, val_idx, dataset, model_type, sample_ids, labels_augmented):
    """Train one fold with comprehensive tracking"""
    global_fold_idx = repeat_idx * N_FOLDS + fold_idx
    fold_seed = SEED + global_fold_idx
    ensure_determinism(fold_seed)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_type.upper()} - Repeat {repeat_idx+1}/{N_REPEATS}, Fold {fold_idx+1}/{N_FOLDS}")
    logger.info(f"Global Fold: {global_fold_idx+1}/{N_REPEATS*N_FOLDS}")
    
    model_class = UnifiedProcessClassifier
    
    if model_type == 'pmi':
        hp = HP_PMI.copy()
        use_pmi = True
    else:
        hp = HP_GEOM.copy()
        use_pmi = False
    
    hp['max_epochs'] = MAX_EPOCHS
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    # DataLoader configuration
    train_loader = DataLoader(
        train_subset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    
    model = model_class(**hp)
    
    checkpoint_dir = OUTPUT_DIR / f"repeat_{repeat_idx}" / f"fold_{fold_idx}" / model_type
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, filename='best', monitor='val_loss', save_top_k=1, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min', verbose=False)
    
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
    
    # Use fixed threshold 0.5 for all classes (matching test set evaluation)
    thresholds = 0.5
    
    # Evaluate
    metrics, preds, labels, probs, cm_per_class, error_analysis, confidences = evaluate_model(model, val_loader, thresholds, use_pmi)
    
    # Store fold metadata
    metrics['repeat'] = repeat_idx
    metrics['fold'] = fold_idx
    metrics['global_fold'] = global_fold_idx
    metrics['epochs_trained'] = trainer.current_epoch + 1
    metrics['model_type'] = model_type
    metrics['threshold'] = 0.5  # Fixed threshold for all classes
    
    # Store confusion matrix details
    for i, class_name in enumerate(CLASS_NAMES):
        tn, fp, fn, tp = cm_per_class[i].ravel()
        metrics[f'cm_{class_name}_tn'] = int(tn)
        metrics[f'cm_{class_name}_fp'] = int(fp)
        metrics[f'cm_{class_name}_fn'] = int(fn)
        metrics[f'cm_{class_name}_tp'] = int(tp)
    
    # Store error patterns
    metrics['error_patterns'] = error_analysis
    
    # Store confidences
    metrics['confidences'] = confidences
    
    # Gate value if PMI
    if use_pmi and hasattr(model, 'gate'):
        metrics['gate_value'] = torch.sigmoid(model.gate).item()
    
    logger.info(f"  F1-Macro: {metrics['f1_macro']:.4f}")
    
    return metrics


def aggregate_cm_stats(results):
    """Aggregate confusion matrix statistics across folds"""
    cm_stats = {}
    
    for class_name in CLASS_NAMES:
        cm_stats[class_name] = {}
        for metric in ['tp', 'fp', 'fn', 'tn']:
            key = f'cm_{class_name}_{metric}'
            values = [r[key] for r in results if key in r]
            cm_stats[class_name][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': int(np.min(values)),
                'max': int(np.max(values)),
                'values': [int(v) for v in values]
            }
    
    return cm_stats


def aggregate_threshold_stats(results):
    """Aggregate threshold statistics"""
    threshold_stats = {
        'fixed': 0.5,
        'note': 'Using fixed threshold 0.5 for all classes to match test set evaluation'
    }
    
    return threshold_stats


def aggregate_error_patterns(results):
    """Aggregate error pattern statistics"""
    # Aggregate error types
    all_error_types = {}
    for r in results:
        if 'error_patterns' in r:
            for error_type, count in r['error_patterns']['error_types'].items():
                if error_type not in all_error_types:
                    all_error_types[error_type] = []
                all_error_types[error_type].append(count)
    
    error_type_stats = {}
    for error_type, counts in all_error_types.items():
        error_type_stats[error_type] = {
            'mean': float(np.mean(counts)),
            'std': float(np.std(counts)),
            'values': counts
        }
    
    # Aggregate specific patterns
    all_patterns = {}
    for r in results:
        if 'error_patterns' in r:
            for pattern, count in r['error_patterns']['specific_patterns'].items():
                if pattern not in all_patterns:
                    all_patterns[pattern] = []
                all_patterns[pattern].append(count)
    
    pattern_stats = {}
    for pattern, counts in all_patterns.items():
        pattern_stats[pattern] = {
            'mean': float(np.mean(counts)),
            'std': float(np.std(counts)),
            'total': int(np.sum(counts)),
            'values': counts
        }
    
    # Sort by total occurrence
    pattern_stats = dict(sorted(pattern_stats.items(), key=lambda x: x[1]['total'], reverse=True))
    
    return {
        'error_types': error_type_stats,
        'specific_patterns': pattern_stats
    }


def aggregate_confidence_stats(results):
    """Aggregate confidence statistics"""
    conf_stats = {}
    
    for class_name in CLASS_NAMES:
        conf_stats[class_name] = {}
        for error_type in ['fn', 'fp']:
            key = f'{class_name}_{error_type}'
            all_confidences = []
            for r in results:
                if 'confidences' in r and key in r['confidences']:
                    all_confidences.extend(r['confidences'][key])
            
            if all_confidences:
                conf_stats[class_name][error_type] = {
                    'mean': float(np.mean(all_confidences)),
                    'std': float(np.std(all_confidences)),
                    'min': float(np.min(all_confidences)),
                    'max': float(np.max(all_confidences)),
                    'count': len(all_confidences),
                    'values': all_confidences
                }
            else:
                conf_stats[class_name][error_type] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'count': 0,
                    'values': []
                }
    
    return conf_stats


def create_extended_visualizations(results_pmi, output_dir):
    """Create extended visualizations for CV analysis"""
    COLORS = {
        'primary': '#2ecc71',
        'secondary': '#e74c3c',
        'tertiary': '#3498db',
        'quaternary': '#f39c12',
        'accent': '#9b59b6'
    }
    
    n_runs = N_FOLDS * N_REPEATS
    
    # 1. Confusion Matrix Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Cross-Validation Results ({N_FOLDS} Folds × {N_REPEATS} Repeats = {n_runs} runs)', 
                 fontsize=14, fontweight='bold')
    
    for idx, class_name in enumerate(CLASS_NAMES):
        data_to_plot = []
        labels = []
        for metric in ['tp', 'fp', 'fn', 'tn']:
            key = f'cm_{class_name}_{metric}'
            values = [r[key] for r in results_pmi if key in r]
            data_to_plot.append(values)
            labels.append(metric.upper())
        
        bp = axes[idx].boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS['primary'])
        axes[idx].set_title(f'{class_name} Confusion Matrix')
        axes[idx].set_ylabel('Count')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cm_boxplots.png', dpi=150)
    plt.close()
    
    # 2. Threshold Distribution (removed - using fixed 0.5)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'Using Fixed Threshold = 0.5\nfor all classes across all folds\n\n(Matches test set evaluation)', 
            ha='center', va='center', fontsize=16, 
            bbox=dict(boxstyle='round', facecolor=COLORS['tertiary'], alpha=0.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'Threshold Strategy ({N_FOLDS} Folds × {N_REPEATS} Repeats)')
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_distribution.png', dpi=150)
    plt.close()
    
    logger.info("Saved extended visualizations")


def analyze_cv_results_extended(results_pmi, output_dir):
    """Extended CV analysis with all tracking data"""
    
    logger.info("\n" + "="*80)
    logger.info("EXTENDED CV ANALYSIS")
    logger.info("="*80)
    
    # Aggregate statistics
    cm_stats = aggregate_cm_stats(results_pmi)
    threshold_stats = aggregate_threshold_stats(results_pmi)
    error_pattern_stats = aggregate_error_patterns(results_pmi)
    confidence_stats = aggregate_confidence_stats(results_pmi)
    
    # Print CM analysis
    logger.info("\n" + "="*80)
    logger.info("CONFUSION MATRIX ANALYSIS (Mean ± Std)")
    logger.info("="*80)
    for class_name in CLASS_NAMES:
        stats = cm_stats[class_name]
        logger.info(f"\n{class_name}:")
        logger.info(f"  TP: {stats['tp']['mean']:.1f} ± {stats['tp']['std']:.1f}  (range: {stats['tp']['min']}-{stats['tp']['max']})")
        logger.info(f"  FP: {stats['fp']['mean']:.1f} ± {stats['fp']['std']:.1f}  (range: {stats['fp']['min']}-{stats['fp']['max']})")
        logger.info(f"  FN: {stats['fn']['mean']:.1f} ± {stats['fn']['std']:.1f}  (range: {stats['fn']['min']}-{stats['fn']['max']})")
        logger.info(f"  TN: {stats['tn']['mean']:.1f} ± {stats['tn']['std']:.1f}  (range: {stats['tn']['min']}-{stats['tn']['max']})")
    
    # Print threshold analysis
    logger.info("\n" + "="*80)
    logger.info("THRESHOLD ANALYSIS")
    logger.info("="*80)
    logger.info("Using fixed threshold = 0.5 for all classes (matches test set evaluation)")
    logger.info("This ensures fair comparison and avoids overfitting to fold-specific thresholds")
    
    # Print error patterns
    logger.info("\n" + "="*80)
    logger.info("ERROR PATTERN ANALYSIS")
    logger.info("="*80)
    logger.info("\nError Types:")
    for error_type, stats in error_pattern_stats['error_types'].items():
        logger.info(f"  {error_type}: {stats['mean']:.1f} ± {stats['std']:.1f}")
    
    logger.info("\nMost Common Specific Patterns (Top 10):")
    for i, (pattern, stats) in enumerate(list(error_pattern_stats['specific_patterns'].items())[:10]):
        logger.info(f"  {i+1}. {pattern}: {stats['mean']:.1f} ± {stats['std']:.1f} (total: {stats['total']})")
    
    # Print confidence analysis
    logger.info("\n" + "="*80)
    logger.info("CONFIDENCE ANALYSIS")
    logger.info("="*80)
    for class_name in CLASS_NAMES:
        logger.info(f"\n{class_name}:")
        fn_stats = confidence_stats[class_name]['fn']
        fp_stats = confidence_stats[class_name]['fp']
        logger.info(f"  FN confidence: {fn_stats['mean']:.3f} ± {fn_stats['std']:.3f}  (n={fn_stats['count']})")
        logger.info(f"  FP confidence: {fp_stats['mean']:.3f} ± {fp_stats['std']:.3f}  (n={fp_stats['count']})")
    
    # Create visualizations
    create_extended_visualizations(results_pmi, output_dir)
    
    # Return aggregated stats
    return {
        'confusion_matrices': cm_stats,
        'thresholds': threshold_stats,
        'error_patterns': error_pattern_stats,
        'confidences': confidence_stats
    }


def run_repeated_cross_validation():
    """Main CV function with extended analysis"""
    logger.info("="*80)
    logger.info("EXTENDED CROSS-VALIDATION")
    logger.info("="*80)
    logger.info(f"Hyperparameters GEOM: {HP_GEOM}")
    logger.info(f"Hyperparameters PMI: {HP_PMI}")
    
    # Load data
    dataset_geom, dataset_pmi, labels, labels_augmented, sample_ids = load_datasets_with_alignment()
    
    results_geom = []
    results_pmi = []
    
    # CV Loop
    for repeat_idx in range(N_REPEATS):
        logger.info(f"\n{'='*80}")
        logger.info(f"REPEAT {repeat_idx + 1}/{N_REPEATS}")
        logger.info("="*80)
        
        repeat_seed = SEED + repeat_idx * 1000
        
        if USE_STRATIFIED:
            splitter = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=repeat_seed)
            split_iter = splitter.split(np.zeros(len(labels_augmented)), labels_augmented)
        else:
            from sklearn.model_selection import KFold
            splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=repeat_seed)
            split_iter = splitter.split(labels)
        
        for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
            for model_type, dataset in [('geom', dataset_geom), ('pmi', dataset_pmi)]:
                metrics = train_fold(repeat_idx, fold_idx, train_idx, val_idx, dataset, model_type, sample_ids, labels_augmented)
                
                if model_type == 'geom':
                    results_geom.append(metrics)
                else:
                    results_pmi.append(metrics)
    
    # Extended analysis
    extended_stats = analyze_cv_results_extended(results_pmi, OUTPUT_DIR)
    
    # Save comprehensive JSON
    final_results = {
        'config': {
            'n_folds': N_FOLDS,
            'n_repeats': N_REPEATS,
            'total_folds': N_FOLDS * N_REPEATS,
            'class_names': CLASS_NAMES,
            'threshold': 0.5,
            'hp_geom': HP_GEOM,
            'hp_pmi': HP_PMI
        },
        'results_pmi': results_pmi,
        'results_geom': results_geom,
        'extended_analysis': extended_stats
    }
    
    with open(OUTPUT_DIR / 'cv_results_extended.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=float)
    
    # Create compact summary (without individual fold results)
    summary_results = {
        'config': final_results['config'],
        'summary': {
            'pmi_model': {
                'f1_macro': {
                    'mean': float(np.mean([r['f1_macro'] for r in results_pmi])),
                    'std': float(np.std([r['f1_macro'] for r in results_pmi])),
                    'min': float(np.min([r['f1_macro'] for r in results_pmi])),
                    'max': float(np.max([r['f1_macro'] for r in results_pmi]))
                },
                'per_class': {}
            },
            'geom_model': {
                'f1_macro': {
                    'mean': float(np.mean([r['f1_macro'] for r in results_geom])),
                    'std': float(np.std([r['f1_macro'] for r in results_geom])),
                    'min': float(np.min([r['f1_macro'] for r in results_geom])),
                    'max': float(np.max([r['f1_macro'] for r in results_geom]))
                },
                'per_class': {}
            }
        },
        'extended_analysis': extended_stats
    }
    
    # Add per-class summaries
    for class_name in CLASS_NAMES:
        for model_type, results in [('pmi_model', results_pmi), ('geom_model', results_geom)]:
            summary_results['summary'][model_type]['per_class'][class_name] = {
                'f1': {
                    'mean': float(np.mean([r[f'f1_{class_name}'] for r in results])),
                    'std': float(np.std([r[f'f1_{class_name}'] for r in results])),
                    'min': float(np.min([r[f'f1_{class_name}'] for r in results])),
                    'max': float(np.max([r[f'f1_{class_name}'] for r in results]))
                },
                'precision': {
                    'mean': float(np.mean([r[f'precision_{class_name}'] for r in results])),
                    'std': float(np.std([r[f'precision_{class_name}'] for r in results]))
                },
                'recall': {
                    'mean': float(np.mean([r[f'recall_{class_name}'] for r in results])),
                    'std': float(np.std([r[f'recall_{class_name}'] for r in results]))
                }
            }
    
    with open(OUTPUT_DIR / 'cv_summary.json', 'w') as f:
        json.dump(summary_results, f, indent=2, default=float)
    
    logger.info(f"\n✓ Extended CV Analysis Complete!")
    logger.info(f"✓ Full results saved to: {OUTPUT_DIR / 'cv_results_extended.json'}")
    logger.info(f"✓ Compact summary saved to: {OUTPUT_DIR / 'cv_summary.json'}")
    
    return final_results


if __name__ == "__main__":
    try:
        results = run_repeated_cross_validation()
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise