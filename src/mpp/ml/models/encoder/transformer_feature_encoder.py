# standard library imports
import logging

# third party imports
import torch
import torch.nn as nn

logging.basicConfig(
    format="%(asctime)s %(levelname)8s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)


class TransformerFeatureEncoder(nn.Module):
    """
    Transformer-based feature encoder for manufacturing process data.
    
    This module extracts feature representations from vector sets (vecsets)
    without performing classification. It's designed to be used as a feature
    extractor in both single-modal and multi-modal architectures.
    
    Architecture:
    - Linear embedding layer
    - Multi-layer Transformer encoder
    - Mean pooling over sequence dimension
    - Output: Feature vector (no classification head)
    
    Parameters
    ----------
    input_dim : int, optional
        Dimensionality of input vectors in the set (default: 32).
    embed_dim : int, optional
        Dimension of the embedding space (default: 128).
    num_heads : int, optional
        Number of attention heads in transformer (default: 8).
    num_layers : int, optional
        Number of transformer encoder layers (default: 3).
    dropout : float, optional
        Dropout rate for regularization (default: 0.1).
    
    Input Shape
    -----------
    x : torch.Tensor
        Shape (batch_size, set_size, input_dim)
        - batch_size: Number of samples in batch
        - set_size: Number of vectors in each set (e.g., 1024 CAD features)
        - input_dim: Dimensionality of each vector (e.g., 32)
    
    Output Shape
    ------------
    features : torch.Tensor
        Shape (batch_size, embed_dim)
        - Feature vector representation for each sample
    
    Examples
    --------
    >>> encoder = TransformerFeatureEncoder(
    ...     input_dim=32,
    ...     embed_dim=128,
    ...     num_heads=8,
    ...     num_layers=3,
    ...     dropout=0.1
    ... )
    >>> vecset = torch.randn(4, 1024, 32)  # batch_size=4, set_size=1024
    >>> features = encoder(vecset)
    >>> print(features.shape)  # torch.Size([4, 128])
    
    Notes
    -----
    - This encoder does NOT include a classification head
    - Mean pooling is used to aggregate sequence information
    - Can be used standalone or as part of larger architectures
    - Designed to work with both geometry-only and multi-modal models
    """
    
    def __init__(
        self,
        input_dim: int = 32,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Store hyperparameters
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer: project input vectors to embedding space
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,  # Input shape: (batch, seq, feature)
            dim_feedforward=embed_dim * 4,  # Standard transformer FF dimension
            activation='gelu'  # GELU activation for better performance
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)  # Final layer normalization
        )
        
        # Layer normalization for output features
        self.output_norm = nn.LayerNorm(embed_dim)
        
        logger.info(f"Initialized TransformerFeatureEncoder: "
                   f"input_dim={input_dim}, embed_dim={embed_dim}, "
                   f"num_heads={num_heads}, num_layers={num_layers}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, set_size, input_dim)
        
        Returns
        -------
        features : torch.Tensor
            Feature representations of shape (batch_size, embed_dim)
        
        Raises
        ------
        AssertionError
            If input tensor has wrong number of dimensions
        """
        # Validate input shape
        assert x.dim() == 3, f"Expected 3D input (batch, seq, features), got shape {x.shape}"
        assert x.size(-1) == self.input_dim, \
            f"Expected input_dim={self.input_dim}, got {x.size(-1)}"
        
        batch_size, set_size, _ = x.shape
        logger.debug(f"Forward pass: input shape {x.shape}")
        
        # Project to embedding space
        x = self.embedding(x)  # [batch, set_size, embed_dim]
        logger.debug(f"After embedding: {x.shape}")
        
        # Pass through transformer encoder
        x = self.encoder(x)  # [batch, set_size, embed_dim]
        logger.debug(f"After transformer: {x.shape}")
        
        # Mean pooling over sequence dimension
        features = x.mean(dim=1)  # [batch, embed_dim]
        logger.debug(f"After pooling: {features.shape}")
        
        # Final normalization
        features = self.output_norm(features)
        
        return features
    
    def get_attention_weights(self, x: torch.Tensor) -> list:
        """
        Extract attention weights from all transformer layers.
        
        Useful for visualization and interpretability.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, set_size, input_dim)
        
        Returns
        -------
        attention_weights : list of torch.Tensor
            List of attention weight tensors, one per layer
            Each tensor has shape (batch_size, num_heads, set_size, set_size)
        """
        x = self.embedding(x)
        
        attention_weights = []
        for layer in self.encoder.layers:
            # Store input for this layer
            layer_input = x
            
            # Get attention weights from self-attention
            # Note: This requires modifying TransformerEncoderLayer to return weights
            # For now, this is a placeholder for future implementation
            x = layer(x)
        
        return attention_weights
    
    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.
        
        Returns
        -------
        int
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Testing and validation
if __name__ == "__main__":
    print("="*60)
    print("Testing TransformerFeatureEncoder")
    print("="*60)
    
    # Test 1: Basic functionality
    print("\nTest 1: Basic forward pass")
    encoder = TransformerFeatureEncoder(
        input_dim=32,
        embed_dim=128,
        num_heads=8,
        num_layers=3,
        dropout=0.1
    )
    
    batch_size = 4
    set_size = 1024
    input_dim = 32
    
    x = torch.randn(batch_size, set_size, input_dim)
    print(f"Input shape: {x.shape}")
    
    features = encoder(x)
    print(f"Output shape: {features.shape}")
    print(f"Expected shape: ({batch_size}, {encoder.embed_dim})")
    assert features.shape == (batch_size, encoder.embed_dim), "Output shape mismatch!"
    print("✓ Test 1 passed")
    
    # Test 2: Different batch sizes
    print("\nTest 2: Variable batch sizes")
    for bs in [1, 8, 16, 32]:
        x = torch.randn(bs, set_size, input_dim)
        features = encoder(x)
        assert features.shape == (bs, encoder.embed_dim)
    print("✓ Test 2 passed")
    
    # Test 3: Parameter count
    print("\nTest 3: Parameter count")
    param_count = encoder.count_parameters()
    print(f"Total trainable parameters: {param_count:,}")
    
    # Test 4: Gradient flow
    print("\nTest 4: Gradient flow")
    x = torch.randn(batch_size, set_size, input_dim, requires_grad=True)
    features = encoder(x)
    loss = features.sum()
    loss.backward()
    assert x.grad is not None, "Gradients not flowing!"
    print("✓ Test 4 passed")
    
    # Test 5: Different configurations
    print("\nTest 5: Different model configurations")
    configs = [
        {"embed_dim": 64, "num_heads": 4, "num_layers": 2},
        {"embed_dim": 256, "num_heads": 16, "num_layers": 6},
        {"embed_dim": 128, "num_heads": 8, "num_layers": 4},
    ]
    
    for config in configs:
        enc = TransformerFeatureEncoder(input_dim=32, **config)
        x = torch.randn(4, 1024, 32)
        out = enc(x)
        print(f"  Config {config}: output shape {out.shape}, params {enc.count_parameters():,}")
    print("✓ Test 5 passed")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)