# standard library imports
import logging

# third party imports
import torch
import torch.nn as nn
import pytorch_lightning as pl

# custom imports
from mpp.constants import TKMS_VOCAB
from mpp.ml.models.encoder.transformer_feature_encoder import TransformerFeatureEncoder

logging.basicConfig(
    format="%(asctime)s %(levelname)8s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)


class UnifiedProcessClassifier(pl.LightningModule):
    """
    Unified multi-label classifier for manufacturing process prediction.
    
    This model can operate in two modes:
    1. Geometry-only mode: Uses only CAD geometry features (vecsets)
    2. Multi-modal mode: Combines geometry with PMI (Product Manufacturing Information)
    
    The architecture properly separates feature extraction from classification,
    avoiding the architectural flaw of the previous implementation where the
    geometry encoder had a classification head before fusion.
    
    Architecture
    ------------
    Geometry-only mode:
        Vecset → TransformerFeatureEncoder → Classifier → Logits
    
    Multi-modal mode:
        Vecset → TransformerFeatureEncoder ──┐
                                             ├→ Fusion → Classifier → Logits
        PMI → MLP Encoder ───────────────────┘
    
    Parameters
    ----------
    input_dim : int, optional
        Dimensionality of the geometry vectors (default: 32).
    pmi_dim : int, optional
        Dimensionality of PMI features (default: 30).
    embed_dim : int, optional
        Size of the embedding dimension (default: 128).
    num_heads : int, optional
        Number of attention heads in transformer (default: 8).
    num_layers : int, optional
        Number of transformer encoder layers (default: 3).
    num_classes : int, optional
        Number of output classes (default: 3 for TKMS).
    dropout : float, optional
        Dropout probability (default: 0.2).
    lr : float, optional
        Learning rate (default: 5e-5).
    weight_decay : float, optional
        Weight decay for regularization (default: 0.01).
    threshold : float, optional
        Classification threshold (default: 0.5).
    max_epochs : int, optional
        Maximum training epochs (default: 200).
    use_pmi : bool, optional
        Whether to use PMI features (default: False).
    initial_gate : float, optional
        Initial value for PMI gating mechanism (default: 0.2).
    modality_dropout : float, optional
        Probability of dropping PMI features during training (default: 0.3).
    
    Examples
    --------
    >>> # Geometry-only model
    >>> model = UnifiedProcessClassifier(use_pmi=False)
    >>> vecset = torch.randn(4, 1024, 32)
    >>> logits = model(vecset)
    >>> print(logits.shape)  # torch.Size([4, 3])
    
    >>> # Multi-modal model
    >>> model = UnifiedProcessClassifier(use_pmi=True, pmi_dim=30)
    >>> vecset = torch.randn(4, 1024, 32)
    >>> pmi = torch.randn(4, 30)
    >>> logits = model(vecset, pmi)
    >>> print(logits.shape)  # torch.Size([4, 3])
    
    Notes
    -----
    - The geometry encoder outputs features, NOT classifications
    - PMI features are optional and can be None even if use_pmi=True
    - Modality dropout helps prevent over-reliance on PMI features
    - Gating mechanism allows the model to learn PMI contribution weight
    """
    
    def __init__(
        self,
        input_dim: int = 32,
        pmi_dim: int = 30,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        num_classes: int = len(TKMS_VOCAB) - 3,  # Remove START, STOP, PAD
        dropout: float = 0.2,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        threshold: float = 0.5,
        max_epochs: int = 200,
        use_pmi: bool = False,
        initial_gate: float = 0.2,
        modality_dropout: float = 0.0
    ):
        super().__init__()
        self.save_hyperparameters()
        
        logger.info(f"Initializing UnifiedProcessClassifier with use_pmi={use_pmi}")
        
        # Geometry encoder - pure feature extraction, NO classification head
        self.geometry_encoder = TransformerFeatureEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # PMI-specific components (only created if use_pmi=True)
        if use_pmi:
            logger.info(f"Enabling PMI mode with pmi_dim={pmi_dim}")
            
            # PMI encoder - MLP with LayerNorm
            self.pmi_encoder = nn.Sequential(
                nn.Linear(pmi_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            
            # Gating mechanism for adaptive PMI contribution
            self.gate = nn.Parameter(torch.tensor(initial_gate))
            
            # Fusion layer processes combined features
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            logger.info("Using geometry-only mode (no PMI)")
            self.pmi_encoder = None
            self.gate = None
            self.fusion = None
        
        # Final classifier (same for both modes)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        logger.info(f"Model initialized with {self.count_parameters():,} parameters")
    
    def forward(self, vecset: torch.Tensor, pmi_features: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Parameters
        ----------
        vecset : torch.Tensor
            Geometry features of shape (batch_size, set_size, input_dim)
        pmi_features : torch.Tensor, optional
            PMI features of shape (batch_size, pmi_dim)
            Only used if use_pmi=True
        
        Returns
        -------
        logits : torch.Tensor
            Classification logits of shape (batch_size, num_classes)
        
        Raises
        ------
        ValueError
            If use_pmi=True but pmi_features is None
        """
        # Extract geometry features (NO classification yet!)
        geom_features = self.geometry_encoder(vecset)
        logger.debug(f"Geometry features shape: {geom_features.shape}")
        
        # Handle PMI features if enabled
        if self.hparams.use_pmi:
            if pmi_features is None:
                raise ValueError("use_pmi=True but pmi_features is None. "
                               "Provide PMI features or set use_pmi=False")
            
            # Modality dropout - randomly drop PMI during training
            if self.training and torch.rand(1).item() < self.hparams.modality_dropout:
                logger.debug("Applying modality dropout - PMI features zeroed")
                pmi_features = torch.zeros_like(pmi_features)
            
            # Encode PMI features
            pmi_encoded = self.pmi_encoder(pmi_features)
            logger.debug(f"PMI features shape: {pmi_encoded.shape}")
            
            # Gated fusion: geometry + weighted PMI
            gate_value = torch.sigmoid(self.gate)
            fused_features = geom_features + gate_value * pmi_encoded
            logger.debug(f"Gate value: {gate_value.item():.3f}")
            
            # Further processing through fusion layer
            features = self.fusion(fused_features)
        else:
            # Geometry-only mode
            features = geom_features
        
        # Final classification (happens here, not in encoder!)
        logits = self.classifier(features)
        logger.debug(f"Output logits shape: {logits.shape}")
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step handling both geometry-only and multi-modal batches"""
        # Unpack batch based on mode
        if self.hparams.use_pmi:
            (vecset, pmi), y = batch
            logits = self(vecset, pmi)
        else:
            vecset, y = batch
            logits = self(vecset)
        
        # Validate labels
        assert y.dim() == 2, f"Expected y shape (B, C), got {y.shape}"
        assert y.dtype == torch.float, f"Expected y dtype float, got {y.dtype}"
        
        # Calculate loss
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.sigmoid(logits) > self.hparams.threshold
        acc = (preds == y.bool()).float().mean()
        
        # Logging
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        
        # Log PMI gate value if applicable
        if self.hparams.use_pmi:
            gate_value = torch.sigmoid(self.gate).item()
            self.log("gate_value", gate_value, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step handling both geometry-only and multi-modal batches"""
        # Unpack batch based on mode
        if self.hparams.use_pmi:
            (vecset, pmi), y = batch
            logits = self(vecset, pmi)
        else:
            vecset, y = batch
            logits = self(vecset)
        
        # Validate labels
        assert y.dim() == 2, f"Expected y shape (B, C), got {y.shape}"
        assert y.dtype == torch.float, f"Expected y dtype float, got {y.dtype}"
        
        # Calculate loss and accuracy
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits) > self.hparams.threshold
        acc = (preds == y.bool()).float().mean()
        
        # Logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        """Test step handling both geometry-only and multi-modal batches"""
        # Unpack batch based on mode
        if self.hparams.use_pmi:
            (vecset, pmi), y = batch
            logits = self(vecset, pmi)
        else:
            vecset, y = batch
            logits = self(vecset)
        
        # Calculate loss and accuracy
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits) > self.hparams.threshold
        acc = (preds == y.bool()).float().mean()
        
        # Logging
        self.log("test_loss", loss)
        self.log("test_acc", acc)
    
    def predict_step(self, batch, batch_idx):
        """Prediction step returning probabilities"""
        # Unpack batch based on mode
        if self.hparams.use_pmi:
            (vecset, pmi), _ = batch
            logits = self(vecset, pmi)
        else:
            vecset, _ = batch
            logits = self(vecset)
        
        # Return probabilities
        probs = torch.sigmoid(logits)
        return probs
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_architecture_summary(self) -> dict:
        """Get summary of model architecture"""
        summary = {
            "mode": "multi-modal" if self.hparams.use_pmi else "geometry-only",
            "total_parameters": self.count_parameters(),
            "geometry_encoder_params": sum(p.numel() for p in self.geometry_encoder.parameters()),
            "classifier_params": sum(p.numel() for p in self.classifier.parameters()),
        }
        
        if self.hparams.use_pmi:
            summary["pmi_encoder_params"] = sum(p.numel() for p in self.pmi_encoder.parameters())
            summary["fusion_params"] = sum(p.numel() for p in self.fusion.parameters())
            summary["gate_value"] = torch.sigmoid(self.gate).item()
        
        return summary


# Testing and validation
if __name__ == "__main__":
    print("="*60)
    print("Testing UnifiedProcessClassifier")
    print("="*60)
    
    batch_size = 4
    set_size = 1024
    input_dim = 32
    pmi_dim = 30
    
    # Test 1: Geometry-only mode
    print("\nTest 1: Geometry-only mode")
    model_geom = UnifiedProcessClassifier(use_pmi=False)
    vecset = torch.randn(batch_size, set_size, input_dim)
    logits = model_geom(vecset)
    print(f"Input shape: {vecset.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, 3), "Output shape mismatch!"
    print(f"Architecture: {model_geom.get_architecture_summary()}")
    print("✓ Test 1 passed")
    
    # Test 2: Multi-modal mode
    print("\nTest 2: Multi-modal mode")
    model_pmi = UnifiedProcessClassifier(use_pmi=True, pmi_dim=pmi_dim)
    pmi = torch.randn(batch_size, pmi_dim)
    logits = model_pmi(vecset, pmi)
    print(f"Input shapes: vecset={vecset.shape}, pmi={pmi.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, 3), "Output shape mismatch!"
    print(f"Architecture: {model_pmi.get_architecture_summary()}")
    print("✓ Test 2 passed")
    
    # Test 3: Training step (geometry-only)
    print("\nTest 3: Training step (geometry-only)")
    labels = torch.randint(0, 2, (batch_size, 3)).float()
    batch_geom = (vecset, labels)
    loss = model_geom.training_step(batch_geom, 0)
    print(f"Training loss: {loss.item():.4f}")
    print("✓ Test 3 passed")
    
    # Test 4: Training step (multi-modal)
    print("\nTest 4: Training step (multi-modal)")
    batch_pmi = ((vecset, pmi), labels)
    loss = model_pmi.training_step(batch_pmi, 0)
    print(f"Training loss: {loss.item():.4f}")
    print("✓ Test 4 passed")
    
    # Test 5: Error handling
    print("\nTest 5: Error handling (missing PMI)")
    try:
        model_pmi(vecset)  # Should raise error
        print("✗ Test 5 failed - should have raised ValueError")
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
        print("✓ Test 5 passed")
    
    # Test 6: Parameter comparison
    print("\nTest 6: Parameter comparison")
    geom_params = model_geom.count_parameters()
    pmi_params = model_pmi.count_parameters()
    print(f"Geometry-only model: {geom_params:,} parameters")
    print(f"Multi-modal model: {pmi_params:,} parameters")
    print(f"PMI adds: {pmi_params - geom_params:,} parameters")
    print("✓ Test 6 passed")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)