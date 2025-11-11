#!/usr/bin/env python3
"""
Extended Unified Training Pipeline with KEY_PMI Support

This pipeline supports geometry-only, full PMI, and KEY PMI training modes
using a single unified model architecture. It includes hyperparameter tuning with
Optuna and baseline training modes.

Usage:
    # Geometry-only training
    python unified_training.py
    
    # Multi-modal training with full PMI (30 features)
    python unified_training.py --use_pmi
    
    # Multi-modal training with KEY PMI only (13 features)
    python unified_training.py --use_key_pmi
    
    # Enable hyperparameter tuning
    python unified_training.py --tune --n_trials 100
    python unified_training.py --use_key_pmi --tune --n_trials 100
"""

# third party imports
import argparse
import optuna
import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import random
from torch.utils.data import Dataset

# custom imports
from mpp.ml.models.classifier.unified_process_classifier import UnifiedProcessClassifier
from mpp.ml.datasets.datamodules import MPP_datamodule
from mpp.ml.datasets.datamodules_pmi import TKMS_PMI_DataModule
from mpp.constants import PATHS

# A40 GPU optimization
torch.set_float32_matmul_precision('medium')

# PMI-specific configuration
PMI_PATH = "/workspace/masterthesis_cadtoplan_fabian_heinze/mpp/encoding_results/standard_encoding.npy"
PMI_CSV_PATH = "/workspace/masterthesis_cadtoplan_fabian_heinze/mpp/data_pmi/pmi_features.csv"
CLIP_VALUE = 5.0

# KEY PMI FEATURES - Based on ablation study results
# Using dimensions (5) + geometric_tolerances (8) = 13 features total
KEY_FEATURES = sorted([
    0, 1, 2, 3, 4,                      # dimensions (5)
    19, 20, 21, 22, 23, 27, 28, 29      # geometric_tolerances (8)
])

# Global variable for experiment directory
CURRENT_EXPERIMENT_DIR = None


class KeyPMIWrapper(Dataset):
    """Wrapper to filter PMI features to only key indices"""
    
    def __init__(self, base_dataset, key_indices=KEY_FEATURES):
        self.base_dataset = base_dataset
        self.key_indices = key_indices
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get original item
        item = self.base_dataset[idx]
        
        # Check if it's PMI data (tuple with PMI features)
        if isinstance(item, tuple) and len(item) == 2:
            data, label = item
            if isinstance(data, tuple) and len(data) == 2:
                vecset, pmi_full = data
                # Filter PMI to key features only
                pmi_key = pmi_full[self.key_indices].float()
                return (vecset, pmi_key), label
        
        return item


class ModelCheckpointWithJSON(ModelCheckpoint):
    """Custom ModelCheckpoint that saves JSON metadata when checkpoint is saved"""
    def __init__(self, model_config=None, **kwargs):
        super().__init__(**kwargs)
        self.model_config = model_config or {}
        
    def _save_checkpoint(self, trainer, filepath):
        """Override to save JSON when checkpoint is actually saved"""
        super()._save_checkpoint(trainer, filepath)
        
        if self.model_config:
            meta_path = Path(filepath).with_suffix('.json')
            
            metadata = {
                "created_at": datetime.now().isoformat(),
                "epoch": trainer.current_epoch,
                "hyperparameters": self.model_config,
                "metrics": {
                    "val_loss": trainer.callback_metrics.get("val_loss", None).item() 
                               if trainer.callback_metrics.get("val_loss") else None,
                    "val_acc": trainer.callback_metrics.get("val_acc", None).item() 
                              if trainer.callback_metrics.get("val_acc") else None,
                }
            }
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            for old_json in Path(filepath).parent.glob("*.json"):
                if old_json != meta_path:
                    old_json.unlink()


def get_dataloaders(batch_size=32, use_pmi=False, use_key_pmi=False):
    """
    Initialize and return dataloaders.
    
    Parameters
    ----------
    batch_size : int
        Batch size for training
    use_pmi : bool
        Whether to use full PMI features (30)
    use_key_pmi : bool
        Whether to use only key PMI features (13)
    
    Returns
    -------
    tuple
        (train_loader, val_loader, pmi_dim)
    """
    if use_pmi or use_key_pmi:
        print(f"Loading PMI datamodule...")
        if use_key_pmi:
            print(f"  Using KEY features only: {len(KEY_FEATURES)} features")
            print(f"  Features: dimensions + geometric_tolerances")
        else:
            print(f"  Using ALL features: 30 features")
            
        dataset = TKMS_PMI_DataModule(
            batch_size=batch_size,
            target_type="step-set",
            pmi_path=PMI_PATH,
            pmi_csv_path=PMI_CSV_PATH,
            clip_value=CLIP_VALUE
        )
        
        dataset.setup(stage="fit")
        
        # Wrap datasets if using key PMI
        if use_key_pmi:
            dataset.train_dataset = KeyPMIWrapper(dataset.train_dataset, KEY_FEATURES)
            dataset.val_dataset = KeyPMIWrapper(dataset.val_dataset, KEY_FEATURES)
            pmi_dim = len(KEY_FEATURES)
        else:
            pmi_dim = 30
            
    else:
        print(f"Loading geometry-only datamodule...")
        dataset = MPP_datamodule(
            batch_size=batch_size,
            input_type="vecset",
            target_type="step-set",
            dataset="tkms"
        )
        dataset.setup(stage="fit")
        pmi_dim = 0
    
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    
    return train_loader, val_loader, pmi_dim


def objective(trial, use_pmi=False, use_key_pmi=False, batch_size=85):
    """
    Optuna objective function for hyperparameter tuning.
    
    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    use_pmi : bool
        Whether to use full PMI features
    use_key_pmi : bool
        Whether to use key PMI features only
    batch_size : int
        Batch size for training
    
    Returns
    -------
    float
        Validation loss to minimize
    """
    trial_id = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Determine mode string
    if use_key_pmi:
        mode_str = "key_pmi"
    elif use_pmi:
        mode_str = "pmi"
    else:
        mode_str = "geom"
    
    # Setup MLflow logging
    experiment_name = f"unified-{mode_str}-tuning"
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri="file:./mlruns",
        run_name=trial_id
    )
    
    # Define hyperparameter search space
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    embed_dim = trial.suggest_categorical("embed_dim", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 5)
    num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    
    if use_pmi or use_key_pmi:
        initial_gate = trial.suggest_float("initial_gate", 0.1, 0.5)
        modality_dropout = trial.suggest_float("modality_dropout", 0.1, 0.5)
    else:
        initial_gate = 0.2
        modality_dropout = 0.3
    
    max_epochs = 50
    
    # Get dataloaders
    train_loader, val_loader, pmi_dim = get_dataloaders(
        batch_size=batch_size, 
        use_pmi=use_pmi,
        use_key_pmi=use_key_pmi
    )
    
    # Initialize unified model
    model = UnifiedProcessClassifier(
        lr=lr,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        use_pmi=(use_pmi or use_key_pmi),
        pmi_dim=pmi_dim,  # Use dynamic pmi_dim
        initial_gate=initial_gate,
        modality_dropout=modality_dropout
    )
    
    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min',
        verbose=True,
    )
    
    trial_dir = CURRENT_EXPERIMENT_DIR / "trials" / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    model_config = {
        "trial_number": trial.number,
        "mode": mode_str,
        "lr": lr,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "use_pmi": use_pmi,
        "use_key_pmi": use_key_pmi,
        "pmi_dim": pmi_dim,
        "max_epochs": max_epochs
    }
    
    if use_pmi or use_key_pmi:
        model_config["pmi_config"] = {
            "path": PMI_PATH,
            "clip_value": CLIP_VALUE,
            "initial_gate": initial_gate,
            "modality_dropout": modality_dropout
        }
        if use_key_pmi:
            model_config["key_features"] = KEY_FEATURES
    
    checkpoint_callback = ModelCheckpointWithJSON(
        model_config=model_config,
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        dirpath=str(trial_dir),
        filename='unified-{epoch:02d}-{val_loss:.4f}',
        save_weights_only=False,
        verbose=True
    )
    
    # Configure trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=mlf_logger,
        enable_checkpointing=True,
        enable_model_summary=False,
        log_every_n_steps=2,
        callbacks=[early_stop_callback, checkpoint_callback],
        precision="16-mixed",
        gradient_clip_val=1.0
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Extract metrics
    val_loss = trainer.callback_metrics["val_loss"].item()
    val_acc = trainer.callback_metrics.get("val_acc", 0.0).item()
    
    # Log to MLflow
    mlf_logger.log_hyperparams(trial.params)
    mlf_logger.log_metrics({
        "val_loss": val_loss,
        "val_acc": val_acc,
        "batch_size": batch_size,
        "pmi_dim": pmi_dim
    })
    
    # Save trial info
    trial_info = {
        "trial_id": trial_id,
        "trial_number": trial.number,
        "model_type": "UnifiedProcessClassifier",
        "use_pmi": use_pmi,
        "use_key_pmi": use_key_pmi,
        "pmi_dim": pmi_dim,
        "hyperparameters": trial.params,
        "batch_size": batch_size,
        "best_checkpoint": checkpoint_callback.best_model_path,
        "metrics": {
            "val_loss": val_loss,
            "val_acc": val_acc,
        },
        "mlflow_run_id": mlf_logger.run_id,
        "epochs_trained": trainer.current_epoch,
    }
    
    if use_key_pmi:
        trial_info["key_features"] = KEY_FEATURES
    
    with open(trial_dir / "trial_config.json", "w") as f:
        json.dump(trial_info, f, indent=4)
    
    return val_loss


def train_baseline(use_pmi=False, use_key_pmi=False, batch_size=85, best_params=None):
    """
    Train baseline model without tuning.
    
    Parameters
    ----------
    use_pmi : bool
        Whether to use full PMI features (30)
    use_key_pmi : bool
        Whether to use key PMI features only (13)
    batch_size : int
        Batch size for training
    best_params : dict, optional
        Best hyperparameters from tuning
    """
    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Get dataloaders with pmi_dim
    train_loader, val_loader, pmi_dim = get_dataloaders(
        batch_size=batch_size,
        use_pmi=use_pmi,
        use_key_pmi=use_key_pmi
    )
    
    # Mode string for naming
    if use_key_pmi:
        mode_str = "key_pmi"
    elif use_pmi:
        mode_str = "pmi"
    else:
        mode_str = "geometry"
    
    # MLflow logging
    mlf_logger = MLFlowLogger(
        experiment_name=f"unified-{mode_str}",
        tracking_uri="file:./mlruns",
        run_name=f"baseline-{mode_str}-{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    
    # Model parameters
    if best_params:
        print(f"\n🏆 Using best parameters from tuning:")
        print(json.dumps(best_params, indent=2))
        model_params = best_params.copy()
        model_params["max_epochs"] = 300
        model_params["use_pmi"] = (use_pmi or use_key_pmi)
        model_params["pmi_dim"] = pmi_dim  # Use dynamic pmi_dim
    else:
        print(f"\n📋 Using default parameters")
        # Default parameters based on mode
        if use_pmi or use_key_pmi:
            # PMI defaults (from your best HP_PMI)
            model_params = {
                "lr": 0.000690,
                "embed_dim": 64,
                "num_layers": 3,
                "num_heads": 8,
                "dropout": 0.280,
                "weight_decay": 0.000277,
                "max_epochs": 300,
                "use_pmi": True,
                "pmi_dim": pmi_dim,  # Dynamic: 13 for key, 30 for full
                "initial_gate": 0.171,
                "modality_dropout": 0.206
            }
        else:
            # Geometry defaults (from your best HP_GEOM)
            model_params = {
                "lr": 0.000326,
                "embed_dim": 128,
                "num_layers": 2,
                "num_heads": 16,
                "dropout": 0.224,
                "weight_decay": 0.000374,
                "max_epochs": 300,
                "use_pmi": False,
                "pmi_dim": 30,  # Still needed for model initialization
                "initial_gate": 0.2,
                "modality_dropout": 0.0
            }
    
    # Initialize model
    model = UnifiedProcessClassifier(**model_params)
    
    # Create checkpoint directory
    training_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    checkpoint_dir = PATHS.CKPT_DIR / "best_model" / f"unified_{mode_str}" / training_timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration for reference
    config_info = {
        "mode": mode_str,
        "use_pmi": use_pmi,
        "use_key_pmi": use_key_pmi,
        "pmi_dim": pmi_dim,
        "batch_size": batch_size,
        "model_params": model_params,
        "created_at": datetime.now().isoformat()
    }
    
    if use_key_pmi:
        config_info["key_features"] = KEY_FEATURES
        config_info["num_key_features"] = len(KEY_FEATURES)
    
    with open(checkpoint_dir / "training_config.json", "w") as f:
        json.dump(config_info, f, indent=4)
    
    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=50,
        mode='min',
        verbose=True,
    )
    
    checkpoint_callback = ModelCheckpointWithJSON(
        model_config=model_params,
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        dirpath=str(checkpoint_dir),
        filename=f'unified_{mode_str}_{{epoch:02d}}_{{val_loss:.4f}}',
        save_weights_only=False,
        verbose=True
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=300,
        logger=mlf_logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_checkpointing=True,
        enable_model_summary=False,
        log_every_n_steps=2,
        precision="16-mixed",
        gradient_clip_val=1.0
    )
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Model: UnifiedProcessClassifier")
    if use_key_pmi:
        print(f"  Mode: KEY PMI (13 features)")
        print(f"  Key features: dimensions + geometric_tolerances")
    elif use_pmi:
        print(f"  Mode: Full PMI (30 features)")
    else:
        print(f"  Mode: Geometry-only")
    print(f"  Dataset: TKMS")
    if use_pmi or use_key_pmi:
        print(f"  PMI path: {PMI_PATH}")
        print(f"  PMI dimension: {pmi_dim}")
        print(f"  PMI clipping: {CLIP_VALUE}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {model_params['max_epochs']}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"{'='*60}\n")
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"  Final val_loss: {trainer.callback_metrics['val_loss']:.4f}")
    print(f"  Final val_acc: {trainer.callback_metrics['val_acc']:.4f}")
    if (use_pmi or use_key_pmi) and hasattr(model, 'gate'):
        print(f"  Final gate value: {torch.sigmoid(model.gate).item():.3f}")
    print(f"{'='*60}\n")


def main():
    global CURRENT_EXPERIMENT_DIR
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Unified training pipeline with KEY PMI support")
    parser.add_argument("--use_pmi", action="store_true", help="Enable full PMI features (30)")
    parser.add_argument("--use_key_pmi", action="store_true", help="Enable KEY PMI features only (13)")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of tuning trials")
    parser.add_argument("--batch_size", type=int, default=85, help="Batch size")
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_pmi and args.use_key_pmi:
        raise ValueError("Cannot use both --use_pmi and --use_key_pmi. Choose one.")
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Determine mode string
    if args.use_key_pmi:
        mode_str = "key_pmi"
        mode_desc = "Key PMI (13 features)"
    elif args.use_pmi:
        mode_str = "pmi"
        mode_desc = "Full PMI (30 features)"
    else:
        mode_str = "geometry"
        mode_desc = "Geometry-only"
    
    mlflow.set_experiment(f"unified-{mode_str}")
    
    print(f"\n{'='*60}")
    print(f"UNIFIED TRAINING PIPELINE")
    print(f"{'='*60}")
    print(f"Mode: {mode_desc}")
    if args.use_key_pmi:
        print(f"Key features: {len(KEY_FEATURES)} (dimensions + geometric_tolerances)")
    print(f"Tuning: {'Enabled' if args.tune else 'Disabled'}")
    if args.tune:
        print(f"Number of trials: {args.n_trials}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*60}\n")
    
    if args.tune:
        # Create experiment directory
        experiment_name = f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_unified_{mode_str}_tuning_{args.n_trials}trials"
        CURRENT_EXPERIMENT_DIR = PATHS.CKPT_DIR / "experiments" / experiment_name
        CURRENT_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save experiment metadata
        experiment_info = {
            "name": experiment_name,
            "description": f"Hyperparameter tuning for unified {mode_desc} model",
            "dataset": "TKMS",
            "mode": mode_str,
            "n_trials": args.n_trials,
            "use_pmi": args.use_pmi,
            "use_key_pmi": args.use_key_pmi,
            "created_at": datetime.now().isoformat(),
        }
        
        if args.use_pmi or args.use_key_pmi:
            experiment_info["pmi_config"] = {
                "path": PMI_PATH,
                "clip_value": CLIP_VALUE
            }
            if args.use_key_pmi:
                experiment_info["key_features"] = KEY_FEATURES
                experiment_info["num_key_features"] = len(KEY_FEATURES)
        
        with open(CURRENT_EXPERIMENT_DIR / "experiment_info.json", "w") as f:
            json.dump(experiment_info, f, indent=4)
        
        # Run hyperparameter tuning
        print("Starting hyperparameter tuning...")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(
                trial, 
                use_pmi=args.use_pmi,
                use_key_pmi=args.use_key_pmi,
                batch_size=args.batch_size
            ),
            n_trials=args.n_trials
        )
        
        best_params = study.best_trial.params
        print(f"\n🏆 Best parameters found:")
        print(json.dumps(best_params, indent=2))
        print(f"Best validation loss: {study.best_value:.4f}")
        
        # Save best parameters
        with open(CURRENT_EXPERIMENT_DIR / "best_params.json", "w") as f:
            json.dump({
                "params": best_params,
                "value": study.best_value,
                "trial_number": study.best_trial.number
            }, f, indent=4)
        
        # Ask if user wants to train with best parameters
        print("\n" + "="*60)
        response = input("Train final model with best parameters? (y/n): ")
        if response.lower() == 'y':
            print("\nTraining final model...")
            train_baseline(
                use_pmi=args.use_pmi,
                use_key_pmi=args.use_key_pmi,
                batch_size=args.batch_size,
                best_params=best_params
            )
    else:
        # Direct baseline training
        train_baseline(
            use_pmi=args.use_pmi,
            use_key_pmi=args.use_key_pmi,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()