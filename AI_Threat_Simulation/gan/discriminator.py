# import torch
# import torch.nn as nn

# class ThreatDiscriminator(nn.Module):
#     def __init__(self, input_dim):
#         super(ThreatDiscriminator, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, data):
#         return self.net(data)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math


class ThreatDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout_rate=0.3, 
                 use_batch_norm=False, use_spectral_norm=True, 
                 activation='leaky_relu', use_sigmoid=False):
        """
        Threat Discriminator for IDS/IPS benchmarking
        
        Args:
            input_dim: Dimension of input threat data
            hidden_dims: List of hidden layer dimensions (auto-computed if None)
            dropout_rate: Dropout probability for regularization
            use_batch_norm: Whether to use batch normalization (usually False for discriminator)
            use_spectral_norm: Whether to use spectral normalization for stability
            activation: Activation function ('relu', 'leaky_relu', 'elu')
            use_sigmoid: Whether to apply sigmoid to output (False for BCEWithLogitsLoss)
        """
        super(ThreatDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_spectral_norm = use_spectral_norm
        self.use_sigmoid = use_sigmoid
        
        # Auto-compute hidden dimensions if not provided
        if hidden_dims is None:
            # Progressive dimension reduction
            hidden_dims = [
                min(512, input_dim * 2),
                min(256, input_dim),
                min(128, input_dim // 2),
                64
            ]
            # Remove dimensions that are too small
            hidden_dims = [dim for dim in hidden_dims if dim >= 32]
        
        self.hidden_dims = hidden_dims
        
        # Build the network layers
        layers = []
        current_dim = input_dim
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer with optional spectral normalization
            linear = nn.Linear(current_dim, hidden_dim)
            if use_spectral_norm:
                linear = spectral_norm(linear)
            layers.append(linear)
            
            # Batch normalization (usually not used in discriminator)
            if use_batch_norm and i > 0:  # Skip BN on first layer
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif activation == 'elu':
                layers.append(nn.ELU(inplace=True))
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            current_dim = hidden_dim
        
        # Output layer
        output_linear = nn.Linear(current_dim, 1)
        if use_spectral_norm:
            output_linear = spectral_norm(output_linear)
        layers.append(output_linear)
        
        # Optional sigmoid activation
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using best practices"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU variants
                nn.init.kaiming_uniform_(module.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, data):
        """
        Forward pass through the discriminator
        
        Args:
            data: Input threat data tensor of shape (batch_size, input_dim)
            
        Returns:
            Discrimination scores (logits if use_sigmoid=False, probabilities if True)
        """
        if data.dim() != 2 or data.size(1) != self.input_dim:
            raise ValueError(f"Expected input of shape (batch_size, {self.input_dim}), "
                           f"got {data.shape}")
        
        return self.net(data)
    
    def get_features(self, data, layer_idx=-2):
        """
        Extract features from a specific layer (useful for analysis)
        
        Args:
            data: Input data
            layer_idx: Index of layer to extract features from (-2 for second-to-last)
        """
        x = data
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == len(self.net) + layer_idx:
                return x
        return x
    
    def get_model_info(self):
        """Get information about the model architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'use_spectral_norm': self.use_spectral_norm,
            'use_sigmoid': self.use_sigmoid
        }


class ConditionalThreatDiscriminator(nn.Module):
    """
    Conditional Discriminator for threat type classification
    Can distinguish between real/fake AND identify threat types
    """
    def __init__(self, input_dim, num_classes, embedding_dim=50, 
                 hidden_dims=None, dropout_rate=0.3, use_spectral_norm=True):
        super(ConditionalThreatDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Embedding layer for threat type conditioning
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        
        # Discriminator network (input is data + embedded class)
        network_input_dim = input_dim + embedding_dim
        
        if hidden_dims is None:
            hidden_dims = [
                min(512, network_input_dim * 2),
                min(256, network_input_dim),
                128,
                64
            ]
        
        layers = []
        current_dim = network_input_dim
        
        for hidden_dim in hidden_dims:
            linear = nn.Linear(current_dim, hidden_dim)
            if use_spectral_norm:
                linear = spectral_norm(linear)
            
            layers.extend([
                linear,
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        # Output layer
        output_linear = nn.Linear(current_dim, 1)
        if use_spectral_norm:
            output_linear = spectral_norm(output_linear)
        layers.append(output_linear)
        
        self.net = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, data, labels):
        """
        Forward pass with threat type conditioning
        
        Args:
            data: Input threat data tensor (batch_size, input_dim)
            labels: Threat type labels (batch_size,)
        """
        # Embed the labels
        embedded_labels = self.embedding(labels)
        
        # Concatenate data and embedded labels
        disc_input = torch.cat([data, embedded_labels], dim=1)
        
        return self.net(disc_input)


class MultiTaskThreatDiscriminator(nn.Module):
    """
    Multi-task discriminator that performs:
    1. Real/Fake classification
    2. Threat type classification
    3. Anomaly scoring
    """
    def __init__(self, input_dim, num_threat_types, hidden_dims=None, 
                 dropout_rate=0.3, use_spectral_norm=True):
        super(MultiTaskThreatDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        self.num_threat_types = num_threat_types
        
        if hidden_dims is None:
            hidden_dims = [min(512, input_dim * 2), min(256, input_dim), 128, 64]
        
        # Shared feature extractor
        shared_layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims[:-1]:  # All but last layer
            linear = nn.Linear(current_dim, hidden_dim)
            if use_spectral_norm:
                linear = spectral_norm(linear)
            
            shared_layers.extend([
                linear,
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        self.shared_features = nn.Sequential(*shared_layers)
        
        # Task-specific heads
        # 1. Real/Fake discrimination
        self.real_fake_head = nn.Sequential(
            nn.Linear(current_dim, hidden_dims[-1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # 2. Threat type classification
        self.threat_type_head = nn.Sequential(
            nn.Linear(current_dim, hidden_dims[-1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dims[-1], num_threat_types)
        )
        
        # 3. Anomaly scoring
        self.anomaly_head = nn.Sequential(
            nn.Linear(current_dim, hidden_dims[-1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()  # Anomaly score between 0 and 1
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, data, return_features=False):
        """
        Forward pass returning multiple outputs
        
        Returns:
            real_fake_logits: Real/fake discrimination logits
            threat_type_logits: Threat type classification logits
            anomaly_scores: Anomaly scores (0-1)
        """
        # Extract shared features
        features = self.shared_features(data)
        
        # Task-specific outputs
        real_fake_logits = self.real_fake_head(features)
        threat_type_logits = self.threat_type_head(features)
        anomaly_scores = self.anomaly_head(features)
        
        if return_features:
            return real_fake_logits, threat_type_logits, anomaly_scores, features
        
        return real_fake_logits, threat_type_logits, anomaly_scores


class WGANThreatDiscriminator(nn.Module):
    """
    Wasserstein GAN Discriminator (Critic) for more stable training
    """
    def __init__(self, input_dim, hidden_dims=None, dropout_rate=0.3):
        super(WGANThreatDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        
        if hidden_dims is None:
            hidden_dims = [min(512, input_dim * 2), min(256, input_dim), 128, 64]
        
        # Build network with spectral normalization (important for WGAN)
        layers = []
        current_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            linear = spectral_norm(nn.Linear(current_dim, hidden_dim))
            layers.append(linear)
            
            if i < len(hidden_dims) - 1:  # No activation on last hidden layer
                layers.extend([
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout_rate)
                ])
            
            current_dim = hidden_dim
        
        # Output layer (no sigmoid for WGAN)
        layers.append(spectral_norm(nn.Linear(current_dim, 1)))
        
        self.net = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, data):
        """
        Forward pass returning Wasserstein distance estimate
        """
        return self.net(data)


def test_discriminators():
    """Test function to verify discriminator functionality"""
    batch_size = 32
    input_dim = 78  # Example: network traffic features
    num_classes = 5  # Example: threat types
    
    # Test basic discriminator
    print("Testing Enhanced ThreatDiscriminator...")
    disc = ThreatDiscriminator(input_dim, use_sigmoid=False)
    data = torch.randn(batch_size, input_dim)
    output = disc(data)
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Model info: {disc.get_model_info()}")
    
    # Test conditional discriminator
    print("\nTesting ConditionalThreatDiscriminator...")
    cdisc = ConditionalThreatDiscriminator(input_dim, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    output = cdisc(data, labels)
    print(f"Conditional output shape: {output.shape}")
    
    # Test multi-task discriminator
    print("\nTesting MultiTaskThreatDiscriminator...")
    mtdisc = MultiTaskThreatDiscriminator(input_dim, num_classes)
    real_fake, threat_type, anomaly = mtdisc(data)
    print(f"Real/Fake output shape: {real_fake.shape}")
    print(f"Threat type output shape: {threat_type.shape}")
    print(f"Anomaly scores shape: {anomaly.shape}")
    
    # Test WGAN discriminator
    print("\nTesting WGANThreatDiscriminator...")
    wdisc = WGANThreatDiscriminator(input_dim)
    output = wdisc(data)
    print(f"WGAN output shape: {output.shape}")
    print(f"WGAN output range: [{output.min():.3f}, {output.max():.3f}]")


if __name__ == "__main__":
    test_discriminators()