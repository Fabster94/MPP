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
        Probability of dropping PMI features during training (default: 0.0).
    pmi_hidden_dim : int, optional
        Hidden dimension for PMI encoder MLP (default: 128).
    pmi_num_layers : int, optional
        Number of hidden layers in PMI encoder (default: 2).
    pmi_dropout : float, optional
        Dropout probability for PMI encoder (default: 0.2).
    fusion_hidden_dim : int, optional
        Hidden dimension for fusion MLP (default: 128).
    fusion_num_layers : int, optional
        Number of layers in fusion network (default: 1).
    fusion_dropout : float, optional
        Dropout probability for fusion network (default: 0.2).
    
    Examples
    --------
    >>> # Geometry-only model
    >>> model = UnifiedProcessClassifier(use_pmi=False)
    >>> vecset = torch.randn(4, 1024, 32)
    >>> logits = model(vecset)
    >>> print(logits.shape)  # torch.Size([4, 3])
    
    >>> # Multi-modal model with custom PMI/Fusion config
    >>> model = UnifiedProcessClassifier(
    ...     use_pmi=True, 
    ...     pmi_dim=30,
    ...     pmi_hidden_dim=256,
    ...     pmi_num_layers=3,
    ...     fusion_hidden_dim=128,
    ...     fusion_num_layers=2
    ... )
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
        modality_dropout: float = 0.0,
        # NEW: PMI Encoder parameters
        pmi_hidden_dim: int = 128,
        pmi_num_layers: int = 2,
        pmi_dropout: float = 0.2,
        # NEW: Fusion parameters
        fusion_hidden_dim: int = 128,
        fusion_num_layers: int = 1,
        fusion_dropout: float = 0.2,
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
            logger.info(f"  PMI Encoder: hidden_dim={pmi_hidden_dim}, num_layers={pmi_num_layers}, dropout={pmi_dropout}")
            logger.info(f"  Fusion: hidden_dim={fusion_hidden_dim}, num_layers={fusion_num_layers}, dropout={fusion_dropout}")
            
            # Build PMI encoder with configurable architecture
            self.pmi_encoder = self._build_pmi_encoder(
                pmi_dim=pmi_dim,
                pmi_hidden_dim=pmi_hidden_dim,
                pmi_num_layers=pmi_num_layers,
                pmi_dropout=pmi_dropout,
                embed_dim=embed_dim
            )
            
            # Gating mechanism for adaptive PMI contribution
            self.gate = nn.Parameter(torch.tensor(initial_gate))
            
            # Build fusion layer with configurable architecture
            self.fusion = self._build_fusion(
                embed_dim=embed_dim,
                fusion_hidden_dim=fusion_hidden_dim,
                fusion_num_layers=fusion_num_layers,
                fusion_dropout=fusion_dropout
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
    
    def _build_pmi_encoder(
        self, 
        pmi_dim: int, 
        pmi_hidden_dim: int, 
        pmi_num_layers: int, 
        pmi_dropout: float, 
        embed_dim: int
    ) -> nn.Sequential:
        """
        Build PMI encoder MLP with configurable depth and width.
        
        Architecture:
        - pmi_num_layers=1: pmi_dim -> pmi_hidden_dim -> embed_dim
        - pmi_num_layers=2: pmi_dim -> pmi_hidden_dim -> pmi_hidden_dim -> embed_dim
        - pmi_num_layers=3: pmi_dim -> pmi_hidden_dim -> pmi_hidden_dim -> pmi_hidden_dim -> embed_dim
        
        Parameters
        ----------
        pmi_dim : int
            Input PMI feature dimension
        pmi_hidden_dim : int
            Hidden layer dimension
        pmi_num_layers : int
            Number of hidden layers (1-3)
        pmi_dropout : float
            Dropout probability
        embed_dim : int
            Output dimension (matches geometry encoder output)
        
        Returns
        -------
        nn.Sequential
            PMI encoder network
        """
        layers = []
        
        # First layer: pmi_dim -> pmi_hidden_dim
        layers.extend([
            nn.Linear(pmi_dim, pmi_hidden_dim),
            nn.LayerNorm(pmi_hidden_dim),
            nn.ReLU(),
            nn.Dropout(pmi_dropout)
        ])
        
        # Additional hidden layers (if pmi_num_layers > 1)
        for _ in range(pmi_num_layers - 1):
            layers.extend([
                nn.Linear(pmi_hidden_dim, pmi_hidden_dim),
                nn.LayerNorm(pmi_hidden_dim),
                nn.ReLU(),
                nn.Dropout(pmi_dropout)
            ])
        
        # Final projection to embed_dim
        layers.extend([
            nn.Linear(pmi_hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        ])
        
        return nn.Sequential(*layers)
    
    def _build_fusion(
        self, 
        embed_dim: int, 
        fusion_hidden_dim: int, 
        fusion_num_layers: int, 
        fusion_dropout: float
    ) -> nn.Sequential:
        """
        Build fusion network with configurable depth and width.
        
        Architecture:
        - fusion_num_layers=1: embed_dim -> embed_dim (simple residual-like)
        - fusion_num_layers=2: embed_dim -> fusion_hidden_dim -> embed_dim
        
        Parameters
        ----------
        embed_dim : int
            Input/output dimension (fused feature dimension)
        fusion_hidden_dim : int
            Hidden layer dimension (used when num_layers > 1)
        fusion_num_layers : int
            Number of layers (1-2)
        fusion_dropout : float
            Dropout probability
        
        Returns
        -------
        nn.Sequential
            Fusion network
        """
        layers = []
        
        if fusion_num_layers == 1:
            # Simple single-layer fusion: embed_dim -> embed_dim
            layers.extend([
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(fusion_dropout)
            ])
        else:
            # Two-layer fusion with hidden dimension: embed_dim -> fusion_hidden_dim -> embed_dim
            layers.extend([
                nn.Linear(embed_dim, fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(fusion_dropout)
            ])
        
        return nn.Sequential(*layers)
    
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
            summary["pmi_config"] = {
                "hidden_dim": self.hparams.pmi_hidden_dim,
                "num_layers": self.hparams.pmi_num_layers,
                "dropout": self.hparams.pmi_dropout
            }
            summary["fusion_params"] = sum(p.numel() for p in self.fusion.parameters())
            summary["fusion_config"] = {
                "hidden_dim": self.hparams.fusion_hidden_dim,
                "num_layers": self.hparams.fusion_num_layers,
                "dropout": self.hparams.fusion_dropout
            }
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
    
    # Test 2: Multi-modal mode (default PMI/Fusion config)
    print("\nTest 2: Multi-modal mode (default config)")
    model_pmi = UnifiedProcessClassifier(use_pmi=True, pmi_dim=pmi_dim)
    pmi = torch.randn(batch_size, pmi_dim)
    logits = model_pmi(vecset, pmi)
    print(f"Input shapes: vecset={vecset.shape}, pmi={pmi.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, 3), "Output shape mismatch!"
    print(f"Architecture: {model_pmi.get_architecture_summary()}")
    print("✓ Test 2 passed")
    
    # Test 3: Multi-modal mode with custom PMI/Fusion config
    print("\nTest 3: Multi-modal mode (custom config)")
    model_custom = UnifiedProcessClassifier(
        use_pmi=True, 
        pmi_dim=pmi_dim,
        pmi_hidden_dim=256,
        pmi_num_layers=3,
        pmi_dropout=0.3,
        fusion_hidden_dim=128,
        fusion_num_layers=2,
        fusion_dropout=0.25
    )
    logits = model_custom(vecset, pmi)
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, 3), "Output shape mismatch!"
    summary = model_custom.get_architecture_summary()
    print(f"PMI config: {summary['pmi_config']}")
    print(f"Fusion config: {summary['fusion_config']}")
    print(f"Total params: {summary['total_parameters']:,}")
    print("✓ Test 3 passed")
    
    # Test 4: Training step (geometry-only)
    print("\nTest 4: Training step (geometry-only)")
    labels = torch.randint(0, 2, (batch_size, 3)).float()
    batch_geom = (vecset, labels)
    loss = model_geom.training_step(batch_geom, 0)
    print(f"Training loss: {loss.item():.4f}")
    print("✓ Test 4 passed")
    
    # Test 5: Training step (multi-modal)
    print("\nTest 5: Training step (multi-modal)")
    batch_pmi = ((vecset, pmi), labels)
    loss = model_pmi.training_step(batch_pmi, 0)
    print(f"Training loss: {loss.item():.4f}")
    print("✓ Test 5 passed")
    
    # Test 6: Error handling
    print("\nTest 6: Error handling (missing PMI)")
    try:
        model_pmi(vecset)  # Should raise error
        print("✗ Test 6 failed - should have raised ValueError")
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
        print("✓ Test 6 passed")
    
    # Test 7: Parameter comparison across configurations
    print("\nTest 7: Parameter comparison")
    configs = [
        {"name": "Geometry-only", "use_pmi": False},
        {"name": "PMI default", "use_pmi": True, "pmi_dim": 30},
        {"name": "PMI small", "use_pmi": True, "pmi_dim": 30, "pmi_hidden_dim": 64, "pmi_num_layers": 1, "fusion_num_layers": 1},
        {"name": "PMI large", "use_pmi": True, "pmi_dim": 30, "pmi_hidden_dim": 256, "pmi_num_layers": 3, "fusion_num_layers": 2},
    ]
    for cfg in configs:
        name = cfg.pop("name")
        model = UnifiedProcessClassifier(**cfg)
        print(f"  {name}: {model.count_parameters():,} parameters")
    print("✓ Test 7 passed")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)