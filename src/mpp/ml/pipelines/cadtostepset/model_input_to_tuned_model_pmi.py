# third party imports
import optuna
import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import random

# custom imports
from mpp.ml.models.classifier.cadtostepset_with_pmi import ProcessClassificationWithPMI
from mpp.ml.datasets.datamodules_pmi import TKMS_PMI_DataModule
from mpp.constants import PATHS


# PMI-specific configuration
PMI_PATH = "encoding_results/standard_encoding.npy"
CLIP_VALUE = 5.0  # Set to 5.0 if you want to clip PMI values / None

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
                "pmi_path": PMI_PATH,
                "pmi_clip_value": CLIP_VALUE,
                "model_type": "ProcessClassificationWithPMI",
                "metrics": {
                    "val_loss": trainer.callback_metrics.get("val_loss", None).item() if trainer.callback_metrics.get("val_loss") else None,
                    "val_acc": trainer.callback_metrics.get("val_acc", None).item() if trainer.callback_metrics.get("val_acc") else None,
                }
            }
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            for old_json in Path(filepath).parent.glob("*.json"):
                if old_json != meta_path:
                    old_json.unlink()

class SimpleMetadataCallback(Callback):
    # Not needed anymore if using ModelCheckpointWithJSON
    pass


def get_dataloaders(batch_size=32):
    """
    Initializes and returns the training and validation dataloaders with PMI.
    """
    dataset = TKMS_PMI_DataModule(
        batch_size=batch_size,
        target_type="step-set",
        pmi_path=PMI_PATH,
        pmi_csv_path= "/workspace/masterthesis_cadtoplan_fabian_heinze/mpp/data_pmi/pmi_features.csv",
        clip_value=CLIP_VALUE
    )
    
    dataset.setup(stage="fit")
    
    train_loader = dataset.train_dataloader()
    validation_loader = dataset.val_dataloader()
    
    # Print PMI statistics
    pmi_stats = dataset.train_dataset.get_pmi_statistics()
    print(f"\nPMI Statistics:")
    print(f"  Shape: {pmi_stats['shape']}")
    print(f"  Range: [{pmi_stats['min']:.3f}, {pmi_stats['max']:.3f}]")
    print(f"  Clipped: {pmi_stats['clipped']}")
    
    return train_loader, validation_loader


# Set batch size
batch_size = 85
train_loader, val_loader = get_dataloaders(batch_size=batch_size)


def objective(trial):
    """Optuna objective for hyperparameter tuning with PMI model"""
    # Generate unique trial ID
    trial_id = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup MLflow logging with trial ID
    mlf_logger = MLFlowLogger(
        experiment_name="cadtostepset-pmi-hyperparameter-tuning",
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
    

    
    # Configuration parameters for tuning
    max_epochs = 50  
    
    # Get dataloaders with trial-specific batch size
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    
    # Initialize PMI model with suggested hyperparameters
    model = ProcessClassificationWithPMI(
        lr=lr,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        pmi_dim=30  # Fixed based on encoding
    )
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min',
        verbose=True,
    )
    
    # Create checkpoint directory with trial info
    trial_dir = CURRENT_EXPERIMENT_DIR / "trials" / f"trial_{trial.number}_pmi"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpointWithJSON(
        model_config={
            "trial_number": trial.number,
            "lr": lr,
            "embed_dim": embed_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "pmi_config": {
                "path": PMI_PATH,
                "clip_value": CLIP_VALUE
        }
        },
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        dirpath=str(trial_dir),
        filename='cadtostepset-pmi-{epoch:02d}-{val_loss:.4f}',
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
        gradient_clip_val=1.0,  # Important for stability with multi-modal
        precision=16  # Mixed precision für schnelleres Training
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Extract metrics
    val_loss = trainer.callback_metrics["val_loss"].item()
    val_acc = trainer.callback_metrics.get("val_acc", 0.0).item()
    
    # Log hyperparameters and metrics to MLflow
    mlf_logger.log_hyperparams(trial.params)
    mlf_logger.log_metrics({
        "val_loss": val_loss,
        "val_acc": val_acc,
        "batch_size": batch_size
    })
    
    # Save complete configuration
    trial_info = {
        "trial_id": trial_id,
        "trial_number": trial.number,
        "model_type": "ProcessClassificationWithPMI",
        "pmi_config": {
            "path": PMI_PATH,
            "clip_value": CLIP_VALUE
        },
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
    
    # Save configuration as JSON in trial directory
    with open(trial_dir / "trial_config.json", "w") as f:
        json.dump(trial_info, f, indent=4)
    
    return val_loss


def main():
    global CURRENT_EXPERIMENT_DIR
    
    # Toggle this to enable/disable hyperparameter tuning
    ENABLE_TUNING = False  # Set to True for hyperparameter tuning
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("classification-pmi")

    if ENABLE_TUNING:
        # Tuning configuration
        N_TRIALS = 100  
        
        # Create experiment directory
        experiment_name = f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_pmi_tuning_{N_TRIALS}trials"
        CURRENT_EXPERIMENT_DIR = PATHS.CKPT_DIR / "experiments" / experiment_name
        CURRENT_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save experiment metadata
        experiment_info = {
            "name": experiment_name,
            "description": "Hyperparameter tuning for PMI-enhanced model on TKMS dataset",
            "dataset": "TKMS with PMI",
            "pmi_path": PMI_PATH,
            "pmi_clip_value": CLIP_VALUE,
            "n_trials": N_TRIALS,
            "created_at": datetime.now().isoformat()
        }
        
        with open(CURRENT_EXPERIMENT_DIR / "experiment_info.json", "w") as f:
            json.dump(experiment_info, f, indent=4)
        
        # Run hyperparameter tuning
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS)
        
        best_params = study.best_trial.params
        print(f"\n🏆 Best parameters found: {best_params}")
        print(f"Best validation loss: {study.best_value:.4f}")
        
    else:

        # Seeds für finales Training setzen
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Für exakte Reproduzierbarkeit
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Normal training without tuning
        mlf_logger = MLFlowLogger(
            experiment_name="classification-pmi",
            tracking_uri="file:./mlruns",
            run_name=f"baseline-pmi-{datetime.now().strftime('%Y%m%d_%H%M')}"
        )

        # Create model with default parameters
        # Beste Parameter aus Trial 45
        model = ProcessClassificationWithPMI(
            lr=0.00175,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1817,
            weight_decay=0.0001587,
            max_epochs=300, #bei trainer gucken
            pmi_dim=30  # Fixed based on encoding
        )
        
        # Timestamp for this training run
        training_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # Create timestamped directory for this training run
        checkpoint_dir = PATHS.CKPT_DIR / "best_model" / "cadtostepset_pmi" / training_timestamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=50,  # Viel Geduld für finales Training
            mode='min',
            verbose=True,
        )

        checkpoint_callback = ModelCheckpointWithJSON(
            model_config=dict(model.hparams),
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            dirpath=str(checkpoint_dir),
            filename='cadtostepset_pmi_{epoch:02d}_val{val_loss:.4f}',
            save_weights_only=False,
            verbose=True
        )



        trainer = Trainer(
            max_epochs=300,
            logger=mlf_logger,
            callbacks=[early_stop_callback, checkpoint_callback],
            enable_checkpointing=True,
            enable_model_summary=False,
            log_every_n_steps=1,
            gradient_clip_val=1.0  # Important for multi-modal stability
        )

        # Print configuration
        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"  Model: ProcessClassificationWithPMI")
        print(f"  Dataset: TKMS with PMI")
        print(f"  PMI path: {PMI_PATH}")
        print(f"  PMI clipping: {CLIP_VALUE}")
        print(f"  Batch size: {batch_size}")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"{'='*60}\n")

        trainer.fit(model, train_loader, val_loader)
        
        # Print final results
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Best checkpoint: {checkpoint_callback.best_model_path}")
        print(f"  Final val_loss: {trainer.callback_metrics['val_loss']:.4f}")
        print(f"  Final val_acc: {trainer.callback_metrics['val_acc']:.4f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()