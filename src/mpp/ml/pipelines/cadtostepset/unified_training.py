#!/usr/bin/env python3
"""
Unified Training Pipeline for Process Classification

This pipeline supports both geometry-only and multi-modal (geometry + PMI) training
using a single unified model architecture. It includes hyperparameter tuning with
Optuna and baseline training modes.

Usage:
    # Geometry-only training
    python unified_training.py
    
    # Multi-modal training with PMI
    python unified_training.py --use_pmi
    
    # Enable hyperparameter tuning
    python unified_training.py --tune --n_trials 100
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

# Global variable for experiment directory
CURRENT_EXPERIMENT_DIR = None


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


def get_dataloaders(batch_size=32, use_pmi=False):
    """
    Initialize and return dataloaders.
    
    Parameters
    ----------
    batch_size : int
        Batch size for training
    use_pmi : bool
        Whether to use PMI features
    
    Returns
    -------
    tuple
        (train_loader, val_loader)
    """
    if use_pmi:
        print(f"Loading PMI datamodule...")
        dataset = TKMS_PMI_DataModule(
            batch_size=batch_size,
            target_type="step-set",
            pmi_path=PMI_PATH,
            pmi_csv_path=PMI_CSV_PATH,
            clip_value=CLIP_VALUE
        )
    else:
        print(f"Loading geometry-only datamodule...")
        dataset = MPP_datamodule(
            batch_size=batch_size,
            input_type="vecset",
            target_type="step-set",
            dataset="tkms"
        )
    
    dataset.setup(stage="fit")
    
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    
    return train_loader, val_loader


def objective(trial, use_pmi=False, batch_size=85):
    """
    Optuna objective function for hyperparameter tuning.
    
    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    use_pmi : bool
        Whether to use PMI features
    batch_size : int
        Batch size for training
    
    Returns
    -------
    float
        Validation loss to minimize
    """
    trial_id = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup MLflow logging
    experiment_name = f"unified-{'pmi' if use_pmi else 'geom'}-tuning"
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
    
    if use_pmi:
        initial_gate = trial.suggest_float("initial_gate", 0.1, 0.5)
        modality_dropout = trial.suggest_float("modality_dropout", 0.1, 0.5)
    else:
        initial_gate = 0.2
        modality_dropout = 0.3
    
    max_epochs = 50
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, use_pmi=use_pmi)
    
    # Initialize unified model
    model = UnifiedProcessClassifier(
        lr=lr,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        use_pmi=use_pmi,
        pmi_dim=30,
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
        "lr": lr,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "use_pmi": use_pmi,
        "max_epochs": max_epochs
    }
    
    if use_pmi:
        model_config["pmi_config"] = {
            "path": PMI_PATH,
            "clip_value": CLIP_VALUE,
            "initial_gate": initial_gate,
            "modality_dropout": modality_dropout
        }
    
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
        "batch_size": batch_size
    })
    
    # Save trial info
    trial_info = {
        "trial_id": trial_id,
        "trial_number": trial.number,
        "model_type": "UnifiedProcessClassifier",
        "use_pmi": use_pmi,
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
    
    with open(trial_dir / "trial_config.json", "w") as f:
        json.dump(trial_info, f, indent=4)
    
    return val_loss


def train_baseline(use_pmi=False, batch_size=85, best_params=None):
    """
    Train baseline model without tuning.
    
    Parameters
    ----------
    use_pmi : bool
        Whether to use PMI features
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
    
    # MLflow logging
    mode_str = "pmi" if use_pmi else "geometry"
    mlf_logger = MLFlowLogger(
        experiment_name=f"unified-{mode_str}",
        tracking_uri="file:./mlruns",
        run_name=f"baseline-{mode_str}-{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, use_pmi=use_pmi)
    
    # Model parameters
    if best_params:
        print(f"\n🏆 Using best parameters from tuning:")
        print(json.dumps(best_params, indent=2))
        model_params = best_params.copy()
        model_params["max_epochs"] = 300
        model_params["use_pmi"] = use_pmi
        model_params["pmi_dim"] = 30
    else:
        print(f"\n📋 Using default parameters")
        # Default parameters (can be adjusted based on your needs)
        model_params = {
            "lr": 0.000806,
            "embed_dim": 64,
            "num_layers": 3,
            "num_heads": 8,
            "dropout": 0.1605,
            "weight_decay": 0.0005109,
            "max_epochs": 300,
            "use_pmi": use_pmi,
            "pmi_dim": 30,
            "initial_gate": 0.2,
            "modality_dropout": 0.3
        }
    
    # Initialize model
    model = UnifiedProcessClassifier(**model_params)
    
    # Create checkpoint directory
    training_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    checkpoint_dir = PATHS.CKPT_DIR / "best_model" / f"unified_{mode_str}" / training_timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
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
    print(f"  Mode: {'Multi-modal (Geometry + PMI)' if use_pmi else 'Geometry-only'}")
    print(f"  Dataset: TKMS")
    if use_pmi:
        print(f"  PMI path: {PMI_PATH}")
        print(f"  PMI clipping: {CLIP_VALUE}")
    print(f"  Batch size: {batch_size}")
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
    if use_pmi and hasattr(model, 'gate'):
        print(f"  Final gate value: {torch.sigmoid(model.gate).item():.3f}")
    print(f"{'='*60}\n")


def main():
    global CURRENT_EXPERIMENT_DIR
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Unified training pipeline")
    parser.add_argument("--use_pmi", action="store_true", help="Enable PMI features")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of tuning trials")
    parser.add_argument("--batch_size", type=int, default=85, help="Batch size")
    args = parser.parse_args()
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mode_str = "pmi" if args.use_pmi else "geometry"
    mlflow.set_experiment(f"unified-{mode_str}")
    
    print(f"\n{'='*60}")
    print(f"UNIFIED TRAINING PIPELINE")
    print(f"{'='*60}")
    print(f"Mode: {'Multi-modal (Geometry + PMI)' if args.use_pmi else 'Geometry-only'}")
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
            "description": f"Hyperparameter tuning for unified {'multi-modal' if args.use_pmi else 'geometry-only'} model",
            "dataset": "TKMS" + (" with PMI" if args.use_pmi else ""),
            "n_trials": args.n_trials,
            "use_pmi": args.use_pmi,
            "created_at": datetime.now().isoformat(),
        }
        
        if args.use_pmi:
            experiment_info["pmi_config"] = {
                "path": PMI_PATH,
                "clip_value": CLIP_VALUE
            }
        
        with open(CURRENT_EXPERIMENT_DIR / "experiment_info.json", "w") as f:
            json.dump(experiment_info, f, indent=4)
        
        # Run hyperparameter tuning
        print("Starting hyperparameter tuning...")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, use_pmi=args.use_pmi, batch_size=args.batch_size),
            n_trials=args.n_trials
        )
        
        best_params = study.best_trial.params
        print(f"\n🏆 Best parameters found:")
        print(json.dumps(best_params, indent=2))
        print(f"Best validation loss: {study.best_value:.4f}")
        
        # Ask if user wants to train with best parameters
        print("\n" + "="*60)
        response = input("Train final model with best parameters? (y/n): ")
        if response.lower() == 'y':
            print("\nTraining final model...")
            train_baseline(use_pmi=args.use_pmi, batch_size=args.batch_size, best_params=best_params)
    else:
        # Direct baseline training
        train_baseline(use_pmi=args.use_pmi, batch_size=args.batch_size)


if __name__ == "__main__":
    main()