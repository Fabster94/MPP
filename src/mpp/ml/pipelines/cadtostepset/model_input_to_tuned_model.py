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

# custom imports
from mpp.ml.models.classifier.cadtostepset import ProcessClassificationTrsfmEncoderModule
from mpp.ml.datasets.datamodules import MPP_datamodule
from mpp.constants import PATHS


class SimpleMetadataCallback(Callback):
    def __init__(self, model_config):
        self.model_config = model_config
        self.last_best_path = None
        
    def on_validation_end(self, trainer, pl_module):
        """Check if best model changed and save metadata only for the best checkpoint"""
        if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
            # Only create JSON if this is a new best model
            if trainer.checkpoint_callback.best_model_path != self.last_best_path:
                self.last_best_path = trainer.checkpoint_callback.best_model_path
                ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
                meta_path = ckpt_path.with_suffix('.json')
                
                # Delete old JSON files (keeps only the best in this training run)
                for old_json in ckpt_path.parent.glob("*.json"):
                    if old_json != meta_path:
                        old_json.unlink()
                
                # Save metadata
                metadata = {
                    "created_at": datetime.now().isoformat(),
                    "epoch": trainer.current_epoch,
                    "hyperparameters": self.model_config,
                    "metrics": {
                        "val_loss": trainer.callback_metrics.get("val_loss", None).item() if trainer.callback_metrics.get("val_loss") else None,
                        "val_acc": trainer.callback_metrics.get("val_acc", None).item() if trainer.callback_metrics.get("val_acc") else None,
                    }
                }
                
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=4)


# ENV-VARS
DATASET_SELECT = "tkms"

# Global variable for experiment directory
CURRENT_EXPERIMENT_DIR = None


def get_dataloaders(batch_size=32, input_type="vecset", target_type="step-set"):
    """
    Initializes and returns the training and validation dataloaders for classification task.

    Parameters
    ----------
    batch_size : int, optional
        Number of samples per batch to load. Default is 32.
    input_type : str, optional
        Input data format type. Default "vecset".
    target_type : str, optional
        Target data format type, e.g. "step-set" for classification labels. Default "step-set".

    Returns
    -------
    tuple of torch.utils.data.DataLoader
        A tuple containing (train_loader, validation_loader).
    """
    dataset = MPP_datamodule(batch_size=batch_size, input_type=input_type, target_type=target_type, dataset=DATASET_SELECT)
    dataset.setup(stage="fit")

    train_loader = dataset.train_dataloader()
    validation_loader = dataset.val_dataloader()

    return train_loader, validation_loader


batch_size = 85
train_loader, val_loader = get_dataloaders(batch_size=batch_size)


def objective(trial):
    # Generate unique trial ID
    trial_id = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup MLflow logging with trial ID
    mlf_logger = MLFlowLogger(
        experiment_name="cadtostepset-hyperparameter-tuning",
        tracking_uri="file:./mlruns",
        run_name=trial_id
    )
    
    # Define hyperparameter search space
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    embed_dim = trial.suggest_categorical("embed_dim", [64, 128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])
    
    # Optional: tune weight_decay as well
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    
    # Configuration parameters for tuning
    max_epochs = 30  # Keep short for tuning phase
    
    # Initialize model with suggested hyperparameters
    model = ProcessClassificationTrsfmEncoderModule(
        lr=lr,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        weight_decay=weight_decay,
        max_epochs=max_epochs
    )
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )
    
    # Create checkpoint directory with trial info
    trial_dir = CURRENT_EXPERIMENT_DIR / "trials" / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        dirpath=str(trial_dir),
        filename='cadtostepset-{epoch:02d}-{val_loss:.4f}',
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
        callbacks=[early_stop_callback, checkpoint_callback]
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
        "val_acc": val_acc
    })
    
    # Save complete configuration
    trial_info = {
        "trial_id": trial_id,
        "trial_number": trial.number,
        "hyperparameters": {
            "lr": lr,
            "embed_dim": embed_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": dropout,
            "weight_decay": weight_decay,
        },
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
    
    return val_loss  # Return value for Optuna optimization


def main():
    global CURRENT_EXPERIMENT_DIR
    
    # Toggle this to enable/disable hyperparameter tuning
    ENABLE_TUNING = False  # Set to True for hyperparameter tuning
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("classification")

    if ENABLE_TUNING:
        # Tuning configuration
        N_TRIALS = 50  # Define as variable
        
        # Try to find current best model automatically
        best_model_dir = PATHS.CKPT_DIR / "best_model/cadtostepset"
        baseline_info = {"model": "None", "val_loss": None}
        
        if best_model_dir.exists():
            # Find the latest checkpoint
            ckpt_files = list(best_model_dir.glob("*.ckpt"))
            if ckpt_files:
                latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
                baseline_info["model"] = latest_ckpt.name
                # Try to extract val_loss from filename
                if "val" in latest_ckpt.name:
                    try:
                        val_loss = float(latest_ckpt.stem.split("val")[-1])
                        baseline_info["val_loss"] = val_loss
                    except:
                        pass
        
        # Create experiment directory only for tuning
        experiment_name = f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_tuning_{N_TRIALS}trials_tkms"
        CURRENT_EXPERIMENT_DIR = PATHS.CKPT_DIR / "experiments" / experiment_name
        CURRENT_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save experiment metadata
        experiment_info = {
            "name": experiment_name,
            "description": "Hyperparameter tuning run on TKMS dataset",
            "dataset": "TKMS", 
            "n_trials": N_TRIALS,
            "created_at": datetime.now().isoformat(),
            "baseline_performance": baseline_info
        }
        
        with open(CURRENT_EXPERIMENT_DIR / "experiment_info.json", "w") as f:
            json.dump(experiment_info, f, indent=4)
        
        # Run hyperparameter tuning
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS)
        
        best_params = study.best_trial.params
        print(f"\n🏆 Best parameters found: {best_params}")
        
    else:
        # Normal training without tuning
        mlf_logger = MLFlowLogger(experiment_name="cadtostepset", tracking_uri="file:./mlruns", run_name="baseline-training")

        # Create model using defaults from cadtostepset.py
        model = ProcessClassificationTrsfmEncoderModule(
        )
        
        # TODO: After baseline documentation, use best parameters from tuning:
        # model = ProcessClassificationTrsfmEncoderModule(
        #     lr=0.000282,
        #     embed_dim=128,
        #     num_layers=4,
        #     num_heads=4,
        #     dropout=0.244,
        #     weight_decay=1.96e-05,
        #     max_epochs=200
        # )

        # Timestamp for this training run
        training_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # Create timestamped directory for this training run
        checkpoint_dir = PATHS.CKPT_DIR / "best_model" / "cadtostepset" / training_timestamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=30,  # Reduced from 100 for faster baseline
            mode='min',
            verbose=True
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            dirpath=str(checkpoint_dir),
            filename='cadtostepset_{epoch:02d}_val{val_loss:.4f}',
            save_weights_only=False,
            verbose=True
        )

        # Create metadata callback with hyperparameters from model
        metadata_callback = SimpleMetadataCallback(dict(model.hparams))

        trainer = Trainer(
            max_epochs=200,
            logger=mlf_logger,
            callbacks=[early_stop_callback, checkpoint_callback, metadata_callback],
            enable_checkpointing=True,
            enable_model_summary=False,
            log_every_n_steps=1
        )

        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
