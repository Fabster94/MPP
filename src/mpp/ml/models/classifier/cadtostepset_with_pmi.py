# standard library imports
import logging

# third party imports
import torch
import torch.nn as nn
import pytorch_lightning as pl

# custom imports
from mpp.constants import TKMS_VOCAB
from mpp.ml.models.classifier.multilabel_classifier import MultilabelTransformerEncoderModule

logging.basicConfig(
    format="%(asctime)s %(levelname)8s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)


class ProcessClassificationWithPMI(pl.LightningModule):
    """
    Multi-modal classifier combining geometry vectors and PMI data.
    
    This model extends the standard classifier to handle both geometry (vecsets)
    and PMI (Product Manufacturing Information) features for improved process prediction.
    
    Architecture:
    - Geometry branch: Transformer encoder (as before)
    - PMI branch: Simple MLP encoder
    - Fusion: Concatenation + Linear layer
    - Output: Multi-label classification
    
    Parameters
    ----------
    input_dim : int, optional
        Dimensionality of the geometry vectors (default: 32).
    pmi_dim : int, optional  
        Dimensionality of PMI features (default: 58).
    embed_dim : int, optional
        Size of the embedding dimension (default: 128).
    num_heads : int, optional
        Number of attention heads (default: 8).
    num_layers : int, optional
        Number of transformer layers (default: 3).
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
    """
    
    def __init__(
        self,
        input_dim=32,
        pmi_dim=32,
        embed_dim=128,
        num_heads=8,
        num_layers=3,
        num_classes=len(TKMS_VOCAB)-3,  # Remove START, STOP, PAD
        dropout=0.2,
        lr=5e-5,
        weight_decay=0.01,
        threshold=0.5,
        max_epochs=200
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Geometry encoder - reuse existing transformer architecture
        self.geometry_encoder = MultilabelTransformerEncoderModule(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=embed_dim,  # Output embedding, not classes
            threshold=threshold
        )
        
        # PMI encoder - simple MLP
        self.pmi_encoder = nn.Sequential(
            nn.Linear(pmi_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim)
        )
        
        # Fusion layer - concatenate and reduce
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final classifier
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, vecset, pmi_features):
        """
        Forward pass combining geometry and PMI features.
        
        Parameters
        ----------
        vecset : torch.Tensor
            Geometry features of shape (batch_size, set_size, input_dim)
        pmi_features : torch.Tensor
            PMI features of shape (batch_size, pmi_dim)
            
        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes)
        """
        # Encode geometry
        geom_features = self.geometry_encoder(vecset)
        
        # Encode PMI
        pmi_features = self.pmi_encoder(pmi_features)
        
        # Fuse features
        combined = torch.cat([geom_features, pmi_features], dim=-1)
        fused = self.fusion(combined)
        
        # Final classification
        logits = self.classifier(fused)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step with multi-modal inputs"""
        (vecset, pmi), y = batch
        
        logits = self(vecset, pmi)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.sigmoid(logits) > self.hparams.threshold
        acc = (preds == y.bool()).float().mean()
        
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with multi-modal inputs"""
        (vecset, pmi), y = batch
        
        logits = self(vecset, pmi)
        loss = self.criterion(logits, y)
        
        preds = torch.sigmoid(logits) > self.hparams.threshold
        acc = (preds == y.bool()).float().mean()
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        """Test step with multi-modal inputs"""
        (vecset, pmi), y = batch
        
        logits = self(vecset, pmi)
        loss = self.criterion(logits, y)
        
        preds = torch.sigmoid(logits) > self.hparams.threshold
        acc = (preds == y.bool()).float().mean()
        
        self.log("test_loss", loss)
        self.log("test_acc", acc)
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
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
    
    def predict_step(self, batch, batch_idx):
        """Prediction step returning probabilities"""
        (vecset, pmi), _ = batch
        
        logits = self(vecset, pmi)
        probs = torch.sigmoid(logits)
        
        return probs


# Minimal test
if __name__ == "__main__":
    # Test model creation
    model = ProcessClassificationWithPMI()
    
    # Test forward pass
    batch_size = 4
    vecset = torch.randn(batch_size, 1024, 32)
    pmi = torch.randn(batch_size, 58)
    
    logits = model(vecset, pmi)
    print(f"Input shapes: vecset={vecset.shape}, pmi={pmi.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")