#!/usr/bin/env python3
"""
Cross-Validation for Geometry vs. Geometry+PMI
Improved version with multilabel stratification, OOF table, and robust evaluation
"""

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    f1_score, accuracy_score, hamming_loss, multilabel_confusion_matrix, jaccard_score,
    precision_recall_fscore_support
)
from torch.utils.data import DataLoader, Subset, ConcatDataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import logging

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

# Custom imports
from mpp.ml.models.classifier.cadtostepset import ProcessClassificationTrsfmEncoderModule
from mpp.ml.models.classifier.cadtostepset_with_pmi import ProcessClassificationWithPMI
from mpp.ml.datasets.tkms import TKMS_Process_Dataset
from mpp.ml.datasets.tkms_pmi import TKMS_PMI_Dataset

# ========== CONFIGURATION ==========
N_FOLDS = 5
SEED = 42
BATCH_SIZE = 85
MAX_EPOCHS = 100
PATIENCE = 20
CLASS_NAMES = ["Bohren", "Drehen", "Fräsen"]
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output Directory
OUTPUT_DIR = Path("cv_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
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
    "lr": 0.000287,
    "embed_dim": 64,
    "num_layers": 3,
    "num_heads": 8,
    "dropout": 0.109,
    "weight_decay": 4.75e-05
}

HP_PMI = {
    "lr": 0.00178,
    "embed_dim": 128,
    "num_layers": 2,
    "num_heads": 16,
    "dropout": 0.338,
    "weight_decay": 1.87e-05,
    "pmi_dim": 58  # Based on your encoded PMI features
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
        logger.warning("Could not enable fully deterministic algorithms. Using standard determinism.")
    logger.info(f"Set seed to {seed} with deterministic settings")


def load_datasets_with_alignment():
    """Load and align datasets ensuring same order"""
    logger.info("Loading and aligning datasets...")

    # Load Train and Valid
    train_geom = TKMS_Process_Dataset(mode="train", target_type="step-set")
    valid_geom = TKMS_Process_Dataset(mode="valid", target_type="step-set")

    train_pmi = TKMS_PMI_Dataset(mode="train", target_type="step-set", **PMI_CONFIG)
    valid_pmi = TKMS_PMI_Dataset(mode="valid", target_type="step-set", **PMI_CONFIG)

    # Verify same sample ordering
    assert train_geom.samples == train_pmi.samples, "Train samples mismatch!"
    assert valid_geom.samples == valid_pmi.samples, "Valid samples mismatch!"

    # Get all sample IDs for reference
    all_sample_ids = train_geom.samples + valid_geom.samples

    # Combine datasets
    dataset_geom = ConcatDataset([train_geom, valid_geom])
    dataset_pmi = ConcatDataset([train_pmi, valid_pmi])

    # Extract labels for stratification
    all_labels = []
    for i in range(len(train_geom) + len(valid_geom)):
        if i < len(train_geom):
            _, label = train_geom[i]
        else:
            _, label = valid_geom[i - len(train_geom)]
        all_labels.append(label.numpy())

    labels = np.array(all_labels)

    # Sanity checks
    assert len(dataset_geom) == len(dataset_pmi) == len(labels), "Dataset size mismatch!"
    assert labels.ndim == 2 and labels.shape[1] == len(CLASS_NAMES), f"Label shape mismatch: {labels.shape}"

    # Check PMI coverage
    pmi_stats = train_pmi.get_pmi_statistics()
    logger.info(f"PMI features: shape={pmi_stats['shape']}, clipped={pmi_stats['clipped']}, clip_value={pmi_stats['clip_value']}")

    logger.info(f"Total samples: {len(labels)}")
    logger.info(f"Label distribution: {labels.sum(axis=0)} ({CLASS_NAMES})")
    logger.info(f"Samples per label count: {np.bincount((labels.sum(axis=1)).astype(int))}")

    return dataset_geom, dataset_pmi, labels, all_sample_ids


def find_optimal_thresholds(model, val_loader, use_pmi=False, per_class=True):
    """Find optimal thresholds - global or per-class"""
    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
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
            best_threshold = 0.5
            best_f1 = 0.0
            for threshold in np.arange(0.1, 0.9, 0.01):
                preds = (all_probs[:, class_idx] > threshold).astype(int)
                f1 = f1_score(all_labels[:, class_idx], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            thresholds[class_idx] = best_threshold
            logger.info(f"  {CLASS_NAMES[class_idx]}: threshold={best_threshold:.2f}, F1={best_f1:.3f}")
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
        logger.info(f"  Global threshold: {best_threshold:.2f}, F1-macro={best_f1:.3f}")

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

            # Apply thresholds
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

    # Calculate metrics
    metrics = {
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'subset_accuracy': accuracy_score(all_labels, all_preds),
        'jaccard_samples': jaccard_score(all_labels, all_preds, average='samples', zero_division=0),
        'jaccard_macro': jaccard_score(all_labels, all_preds, average='macro', zero_division=0),
        'hamming_loss': hamming_loss(all_labels, all_preds),
    }

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    for i, class_name in enumerate(CLASS_NAMES):
        metrics[f'f1_{class_name}'] = f1[i]
        metrics[f'precision_{class_name}'] = precision[i]
        metrics[f'recall_{class_name}'] = recall[i]
        metrics[f'support_{class_name}'] = support[i]

    # Store thresholds
    if isinstance(thresholds, np.ndarray):
        for i, class_name in enumerate(CLASS_NAMES):
            metrics[f'threshold_{class_name}'] = thresholds[i]
    else:
        metrics['threshold_global'] = thresholds

    # Confusion matrices
    cm_per_class = multilabel_confusion_matrix(all_labels, all_preds)

    return metrics, all_preds, all_labels, all_probs, cm_per_class


def train_fold(fold_idx, train_idx, val_idx, dataset, model_type, sample_ids):
    """Train one fold with improved tracking"""
    ensure_determinism(SEED + fold_idx)

    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_type.upper()} - Fold {fold_idx + 1}/{N_FOLDS}")
    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Choose model and hyperparameters
    if model_type == 'pmi':
        model_class = ProcessClassificationWithPMI
        hp = HP_PMI.copy()
        use_pmi = True
    else:
        model_class = ProcessClassificationTrsfmEncoderModule
        hp = HP_GEOM.copy()
        use_pmi = False

    hp['max_epochs'] = MAX_EPOCHS

    # Create dataloaders (robust when NUM_WORKERS=0)
    PERSIST = NUM_WORKERS > 0

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=PERSIST,
        prefetch_factor=2 if PERSIST else None
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=PERSIST,
        prefetch_factor=2 if PERSIST else None
    )

    # Initialize model
    model = model_class(**hp)

    # Callbacks
    checkpoint_dir = OUTPUT_DIR / f"fold_{fold_idx}" / model_type
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
        verbose=True
    )

    # Trainer
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stop],
        enable_progress_bar=True,
        logger=False,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        gradient_clip_val=1.0,
        deterministic=True
    )

    # Training
    trainer.fit(model, train_loader, val_loader)

    # Load best model
    model = model_class.load_from_checkpoint(checkpoint_callback.best_model_path)
    model.to(DEVICE)
    model.eval()

    # Find optimal thresholds
    logger.info("Finding optimal thresholds...")
    thresholds = find_optimal_thresholds(model, val_loader, use_pmi, per_class=True)

    # Evaluate
    metrics, preds, labels, probs, cm = evaluate_model(model, val_loader, thresholds, use_pmi)

    # Add fold metadata
    metrics['fold'] = fold_idx
    metrics['epochs_trained'] = trainer.current_epoch + 1  # Anzahl Epochen
    metrics['best_model_path'] = str(checkpoint_callback.best_model_path)
    metrics['early_stopped_epoch'] = trainer.current_epoch + 1
    metrics['model_type'] = model_type

    # Save fold data
    val_sample_ids = [sample_ids[i] for i in val_idx]

    fold_results = {
        'metrics': metrics,
        'val_sample_ids': val_sample_ids,
        'val_indices': val_idx.tolist(),
        'checkpoint_path': str(checkpoint_callback.best_model_path),
        'hyperparameters': hp
    }

    # Save all outputs
    np.save(checkpoint_dir / "predictions.npy", preds)
    np.save(checkpoint_dir / "labels.npy", labels)
    np.save(checkpoint_dir / "probabilities.npy", probs)
    np.save(checkpoint_dir / "confusion_matrices.npy", cm)
    np.save(checkpoint_dir / "thresholds.npy", thresholds)

    with open(checkpoint_dir / "fold_results.json", 'w') as f:
        json.dump(fold_results, f, indent=4, default=str)

    logger.info(f"  F1-Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"  F1-Micro: {metrics['f1_micro']:.4f}")
    logger.info(f"  Jaccard-Samples: {metrics['jaccard_samples']:.4f}")

    return metrics, preds, labels, cm


def build_oof_table(output_dir: Path) -> pd.DataFrame:
    """Build OOF predictions table from all fold results"""
    rows = []
    for fold_root in sorted(output_dir.glob("fold_*")):
        for model_dir in [fold_root / "geom", fold_root / "pmi"]:
            if not model_dir.is_dir():
                continue

            needed = ["predictions.npy", "labels.npy", "probabilities.npy", "thresholds.npy", "fold_results.json"]
            if any(not (model_dir / f).exists() for f in needed):
                logger.warning("Skipping %s (missing files)", model_dir)
                continue

            # Load required files
            preds = np.load(model_dir / "predictions.npy")
            labels = np.load(model_dir / "labels.npy")
            probs = np.load(model_dir / "probabilities.npy")
            thresholds = np.load(model_dir / "thresholds.npy", allow_pickle=True)
            with open(model_dir / "fold_results.json") as f:
                meta = json.load(f)

            ids = meta["val_sample_ids"]
            model_type = meta["metrics"]["model_type"]
            fold = meta["metrics"]["fold"]

            df = pd.DataFrame({"id": ids, "fold": fold, "model_type": model_type})
            for i, cls in enumerate(CLASS_NAMES):
                df[f"y_{cls}"] = labels[:, i].astype(int)
                df[f"p_{cls}"] = probs[:, i]
                df[f"hat_{cls}"] = preds[:, i].astype(int)

            # thresholds
            if isinstance(thresholds, np.ndarray):
                for i, cls in enumerate(CLASS_NAMES):
                    df[f"thr_{cls}"] = float(thresholds[i])
            else:
                df["thr_global"] = float(thresholds)

            rows.append(df)

    oof_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    if not oof_df.empty:
        logger.info(f"Saved OOF predictions to {output_dir / 'oof_predictions.csv'} ({len(oof_df)} rows)")
        for model_type in ['geom', 'pmi']:
            model_data = oof_df[oof_df['model_type'] == model_type]
            if len(model_data) > 0:
                y_true = model_data[[f"y_{c}" for c in CLASS_NAMES]].values
                y_pred = model_data[[f"hat_{c}" for c in CLASS_NAMES]].values
                oof_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
                oof_f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
                oof_jaccard = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
                logger.info(f"\nOOF Metrics - {model_type.upper()}:")
                logger.info(f"  F1-Macro: {oof_f1_macro:.4f}")
                logger.info(f"  F1-Micro: {oof_f1_micro:.4f}")
                logger.info(f"  Jaccard-Samples: {oof_jaccard:.4f}")

    return oof_df


def create_comprehensive_plots(df_geom, df_pmi, output_dir):
    """Create comprehensive visualization plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Box plot comparison
    ax = axes[0, 0]
    data = pd.DataFrame({
        'Geometry': df_geom['f1_macro'],
        'Geometry+PMI': df_pmi['f1_macro']
    })
    data.boxplot(ax=ax)
    ax.set_ylabel('F1-Macro Score')
    ax.set_title('F1-Macro Distribution')
    ax.grid(True, alpha=0.3)

    # 2. Paired differences plot
    ax = axes[0, 1]
    differences = df_pmi['f1_macro'].values - df_geom['f1_macro'].values
    x = np.arange(len(differences))
    colors = ['green' if d > 0 else 'red' for d in differences]
    ax.bar(x, differences, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Fold')
    ax.set_ylabel('F1-Macro Difference (PMI - Geom)')
    ax.set_title('Per-Fold Improvement')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(len(differences))])

    # 3. Per-class comparison
    ax = axes[0, 2]
    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    means_geom = [df_geom[f'f1_{c}'].mean() for c in CLASS_NAMES]
    means_pmi = [df_pmi[f'f1_{c}'].mean() for c in CLASS_NAMES]
    stds_geom = [df_geom[f'f1_{c}'].std() for c in CLASS_NAMES]
    stds_pmi = [df_pmi[f'f1_{c}'].std() for c in CLASS_NAMES]

    ax.bar(x - width/2, means_geom, width, yerr=stds_geom, label='Geometry', alpha=0.8, capsize=5)
    ax.bar(x + width/2, means_pmi, width, yerr=stds_pmi, label='Geometry+PMI', alpha=0.8, capsize=5)

    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Process Class')
    ax.set_title('Per-Class Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Multiple metrics comparison
    ax = axes[1, 0]
    metrics = ['f1_macro', 'f1_micro', 'jaccard_samples', 'subset_accuracy']
    metric_labels = ['F1-Macro', 'F1-Micro', 'Jaccard-Samples', 'Subset Acc.\n(exact match)']

    geom_means = [df_geom[m].mean() for m in metrics]
    pmi_means = [df_pmi[m].mean() for m in metrics]

    x = np.arange(len(metrics))
    ax.bar(x - width/2, geom_means, width, label='Geometry', alpha=0.8)
    ax.bar(x + width/2, pmi_means, width, label='Geometry+PMI', alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('Multiple Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Learning curves (epochs)
    ax = axes[1, 1]
    ax.scatter(df_geom['epochs_trained'], df_geom['f1_macro'], label='Geometry', alpha=0.7, s=100)
    ax.scatter(df_pmi['epochs_trained'], df_pmi['f1_macro'], label='Geometry+PMI', alpha=0.7, s=100)
    ax.set_xlabel('Epochs Trained')
    ax.set_ylabel('F1-Macro Score')
    ax.set_title('Training Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Improvement heatmap
    ax = axes[1, 2]
    improvements = pd.DataFrame({
        'F1-Macro': [((df_pmi['f1_macro'].mean() - df_geom['f1_macro'].mean()) / df_geom['f1_macro'].mean()) * 100],
        'F1-Micro': [((df_pmi['f1_micro'].mean() - df_geom['f1_micro'].mean()) / df_geom['f1_micro'].mean()) * 100],
        'Jaccard': [((df_pmi['jaccard_samples'].mean() - df_geom['jaccard_samples'].mean()) / df_geom['jaccard_samples'].mean()) * 100],
        'Subset Acc': [((df_pmi['subset_accuracy'].mean() - df_geom['subset_accuracy'].mean()) / df_geom['subset_accuracy'].mean()) * 100]
    })

    sns.heatmap(improvements, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Improvement %'}, ax=ax)
    ax.set_title('Relative Improvements with PMI')
    ax.set_yticklabels(['PMI vs Geom'], rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'cv_results_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.close()


def analyze_results(results):
    """Analyze and visualize CV results with statistical tests"""
    logger.info("\n" + "="*80)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("="*80)

    # Convert to DataFrame
    df_geom = pd.DataFrame(results['geom'])
    df_pmi = pd.DataFrame(results['pmi'])

    # Calculate statistics
    metrics_to_analyze = ['f1_macro', 'f1_micro', 'jaccard_samples', 'subset_accuracy', 'hamming_loss']

    logger.info("\nNote: 'subset_accuracy' ist sehr strikt (Exact Match). 'jaccard_samples' ist oft aussagekräftiger.\n")

    for model_type, df in [('Geometry-only', df_geom), ('Geometry+PMI', df_pmi)]:
        logger.info(f"\n{model_type}:")
        for metric in metrics_to_analyze:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            logger.info(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

        for class_name in CLASS_NAMES:
            f1_key = f'f1_{class_name}'
            if f1_key in df.columns:
                mean_f1 = df[f1_key].mean()
                std_f1 = df[f1_key].std()
                logger.info(f"  F1-{class_name}: {mean_f1:.4f} ± {std_f1:.4f}")

    # Improvements and Wilcoxon
    logger.info("\nImprovements (PMI vs Geometry):")
    test_results = {}
    for metric in ['f1_macro', 'f1_micro', 'jaccard_samples']:
        geom_mean = df_geom[metric].mean()
        pmi_mean = df_pmi[metric].mean()
        improvement = ((pmi_mean - geom_mean) / max(1e-12, geom_mean)) * 100
        logger.info(f"  {metric}: {improvement:+.1f}%")
        if len(df_geom) >= 5:
            stat, p_value = wilcoxon(df_pmi[metric], df_geom[metric])
            logger.info(f"    Wilcoxon: statistic={stat:.4f}, p={p_value:.4f} → {'sig.' if p_value < 0.05 else 'n.s.'}")
            test_results[metric] = {'statistic': float(stat), 'p_value': float(p_value)}

    logger.info("\nPer-class F1 (Wilcoxon):")
    for class_name in CLASS_NAMES:
        key = f'f1_{class_name}'
        if key in df_geom.columns and len(df_geom) >= 5:
            stat, p_value = wilcoxon(df_pmi[key], df_geom[key])
            logger.info(f"  {class_name}: p={p_value:.4f} → {'sig.' if p_value < 0.05 else 'n.s.'}")
            test_results[key] = {'statistic': float(stat), 'p_value': float(p_value)}

    # Plots
    create_comprehensive_plots(df_geom, df_pmi, OUTPUT_DIR)

    # Save detailed results
    summary = {
        'geom': {
            metric: {
                'mean': df_geom[metric].mean(),
                'std': df_geom[metric].std(),
                'values': df_geom[metric].tolist()
            }
            for metric in df_geom.columns if metric not in ['fold', 'model_type', 'epochs_trained', 'best_model_path']
        },
        'pmi': {
            metric: {
                'mean': df_pmi[metric].mean(),
                'std': df_pmi[metric].std(),
                'values': df_pmi[metric].tolist()
            }
            for metric in df_pmi.columns if metric not in ['fold', 'model_type', 'epochs_trained', 'best_model_path']
        },
        'statistical_tests': test_results
    }

    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

    # Save DataFrames
    df_geom.to_csv(OUTPUT_DIR / 'results_geom.csv', index=False)
    df_pmi.to_csv(OUTPUT_DIR / 'results_pmi.csv', index=False)


def run_cross_validation():
    """Main cross-validation function with improvements"""
    logger.info("="*80)
    logger.info("CROSS-VALIDATION: Geometry vs. Geometry+PMI")
    logger.info(f"Configuration: {N_FOLDS} folds, Seed: {SEED}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("="*80)

    # Save configuration
    config = {
        'n_folds': N_FOLDS,
        'seed': SEED,
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'patience': PATIENCE,
        'hp_geom': HP_GEOM,
        'hp_pmi': HP_PMI,
        'pmi_config': PMI_CONFIG,
        'class_names': CLASS_NAMES
    }

    with open(OUTPUT_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # Load aligned data
    dataset_geom, dataset_pmi, labels, sample_ids = load_datasets_with_alignment()

    # Store all results
    results = {'geom': [], 'pmi': []}

    # Multilabel Stratified K-Fold
    if USE_STRATIFIED:
        mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        split_iter = mskf.split(labels, labels)
    else:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        split_iter = kf.split(labels)

    # Cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        logger.info(f"\n{'='*80}")
        logger.info(f"FOLD {fold_idx + 1}/{N_FOLDS}")
        logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

        # Check label distribution in fold
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        logger.info(f"Train label dist: {train_labels.sum(axis=0)}")
        logger.info(f"Val label dist: {val_labels.sum(axis=0)}")

        # Train both models on same fold
        for model_type, dataset in [('geom', dataset_geom), ('pmi', dataset_pmi)]:
            metrics, preds, labels_fold, cm = train_fold(
                fold_idx, train_idx, val_idx, dataset, model_type, sample_ids
            )
            results[model_type].append(metrics)

    # Build OOF table after all folds are complete
    oof_df = build_oof_table(OUTPUT_DIR)

    # Analyze and visualize results
    analyze_results(results)

    return results, oof_df


if __name__ == "__main__":
    try:
        results, oof_df = run_cross_validation()
        logger.info(f"\n✓ Cross-Validation completed successfully!")
        logger.info(f"✓ Results saved in: {OUTPUT_DIR}")
        logger.info(f"✓ OOF predictions available for {len(oof_df)} rows")
    except Exception as e:
        logger.error(f"Error during cross-validation: {str(e)}", exc_info=True)
        raise