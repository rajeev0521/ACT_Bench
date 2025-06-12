# import torch
# import torch.nn as nn

# class ThreatGenerator(nn.Module):
#     def __init__(self, noise_dim, output_dim):
#         super(ThreatGenerator, self).__init__()
#         self.noise_dim = noise_dim
#         self.output_dim = output_dim
#         self.net = nn.Sequential(
#             nn.Linear(noise_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, output_dim),
#             nn.Tanh()
#         )

#     def forward(self, noise):
#         return self.net(noise)



import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ThreatGenerator(nn.Module):
    def __init__(self, noise_dim, output_dim, hidden_dims=None, dropout_rate=0.2, 
                 use_batch_norm=True, activation='leaky_relu', output_activation='tanh'):
        """
        Enhanced Threat Generator for IDS/IPS benchmarking
        
        Args:
            noise_dim: Dimension of input noise vector
            output_dim: Dimension of output threat data
            hidden_dims: List of hidden layer dimensions (auto-computed if None)
            dropout_rate: Dropout probability for regularization
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'leaky_relu', 'elu')
            output_activation: Final layer activation ('tanh', 'sigmoid', 'none')
        """
        super(ThreatGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Auto-compute hidden dimensions if not provided
        if hidden_dims is None:
            # Progressive expansion then contraction
            max_dim = max(256, output_dim * 2)
            hidden_dims = [
                max_dim,
                max_dim // 2,
                max_dim // 4,
                output_dim * 2 if output_dim > 32 else 64
            ]
        
        self.hidden_dims = hidden_dims
        
        # Build the network layers
        layers = []
        input_dim = noise_dim
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif activation == 'elu':
                layers.append(nn.ELU(inplace=True))
            
            # Dropout (not on last hidden layer)
            if dropout_rate > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))
            
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, output_dim))
        
        # Output activation
        if output_activation == 'tanh':
            layers.append(nn.Tanh())
        elif output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        # 'none' case: no activation
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using best practices"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, noise):
        """
        Forward pass through the generator
        
        Args:
            noise: Input noise tensor of shape (batch_size, noise_dim)
            
        Returns:
            Generated threat data of shape (batch_size, output_dim)
        """
        if noise.dim() != 2 or noise.size(1) != self.noise_dim:
            raise ValueError(f"Expected noise of shape (batch_size, {self.noise_dim}), "
                           f"got {noise.shape}")
        
        return self.net(noise)
    
    def generate_samples(self, n_samples, device='cpu'):
        """
        Generate synthetic threat samples
        
        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated samples tensor
        """
        self.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.noise_dim, device=device)
            samples = self.forward(noise)
        return samples
    
    def get_model_info(self):
        """Get information about the model architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'noise_dim': self.noise_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm
        }


class ConditionalThreatGenerator(nn.Module):
    """
    Conditional GAN Generator for specific threat types
    Useful for generating targeted attack patterns
    """
    def __init__(self, noise_dim, output_dim, num_classes, embedding_dim=50, 
                 hidden_dims=None, dropout_rate=0.2):
        super(ConditionalThreatGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Embedding layer for threat type conditioning
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        
        # Generator network (input is noise + embedded class)
        input_dim = noise_dim + embedding_dim
        
        if hidden_dims is None:
            max_dim = max(256, output_dim * 2)
            hidden_dims = [max_dim, max_dim // 2, max_dim // 4, output_dim * 2]
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.extend([
            nn.Linear(input_dim, output_dim),
            nn.Tanh()
        ])
        
        self.net = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, noise, labels):
        """
        Forward pass with threat type conditioning
        
        Args:
            noise: Input noise tensor (batch_size, noise_dim)
            labels: Threat type labels (batch_size,)
        """
        # Embed the labels
        embedded_labels = self.embedding(labels)
        
        # Concatenate noise and embedded labels
        gen_input = torch.cat([noise, embedded_labels], dim=1)
        
        return self.net(gen_input)


class ResidualThreatGenerator(nn.Module):
    """
    Generator with residual connections for better gradient flow
    Particularly useful for high-dimensional network data
    """
    def __init__(self, noise_dim, output_dim, num_blocks=3, hidden_dim=256, 
                 dropout_rate=0.2):
        super(ResidualThreatGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        
        # Initial projection
        self.input_proj = nn.Linear(noise_dim, hidden_dim)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._make_residual_block(hidden_dim, dropout_rate) 
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _make_residual_block(self, dim, dropout_rate):
        """Create a residual block"""
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, noise):
        x = self.input_proj(noise)
        
        # Apply residual blocks
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection
            x = F.leaky_relu(x, 0.2, inplace=True)
        
        return self.output_proj(x)


def test_generators():
    """Test function to verify generator functionality"""
    batch_size = 32
    noise_dim = 100
    output_dim = 78  # Example: NSL-KDD dataset features
    num_classes = 5  # Example: Normal, DoS, Probe, R2L, U2R
    
    # Test basic generator
    print("Testing Basic ThreatGenerator...")
    gen = ThreatGenerator(noise_dim, output_dim)
    noise = torch.randn(batch_size, noise_dim)
    output = gen(noise)
    print(f"Input shape: {noise.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Model info: {gen.get_model_info()}")
    
    # Test conditional generator
    print("\nTesting ConditionalThreatGenerator...")
    cgen = ConditionalThreatGenerator(noise_dim, output_dim, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    output = cgen(noise, labels)
    print(f"Conditional output shape: {output.shape}")
    
    # Test residual generator
    print("\nTesting ResidualThreatGenerator...")
    rgen = ResidualThreatGenerator(noise_dim, output_dim)
    output = rgen(noise)
    print(f"Residual output shape: {output.shape}")


if __name__ == "__main__":
    test_generators()