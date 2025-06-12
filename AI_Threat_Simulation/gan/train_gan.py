import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import logging
from datetime import datetime

from generator import ThreatGenerator
from discriminator import ThreatDiscriminator


class CSVDataset(Dataset):
    def __init__(self, csv_path, normalize=True, categorical_cols=None):
        """
        Enhanced dataset class with normalization and categorical handling
        """
        self.data_raw = pd.read_csv(csv_path)
        
        # Handle categorical columns if specified
        if categorical_cols:
            self.label_encoders = {}
            for col in categorical_cols:
                if col in self.data_raw.columns:
                    le = LabelEncoder()
                    self.data_raw[col] = le.fit_transform(self.data_raw[col].astype(str))
                    self.label_encoders[col] = le
        
        # Normalize data if requested
        if normalize:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data_raw.values)
        else:
            self.scaler = None
            self.data = self.data_raw.values
            
        self.data = torch.tensor(self.data, dtype=torch.float32)
        
        # Store metadata
        self.feature_names = list(self.data_raw.columns)
        self.n_features = len(self.feature_names)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def inverse_transform(self, data):
        """Convert normalized data back to original scale"""
        if self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data


def setup_logging():
    """Setup logging for training monitoring"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def infer_dims(csv_path, categorical_cols=None):
    """Infer dataset dimensions dynamically with enhanced metadata."""
    dataset = CSVDataset(csv_path, categorical_cols=categorical_cols)
    output_dim = dataset[0].shape[0]
    noise_dim = max(64, output_dim)  # Increased noise dimension for better generation
    
    return output_dim, noise_dim, dataset


def save_training_config(config, model_dir):
    """Save training configuration for reproducibility"""
    config_path = os.path.join(model_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def plot_losses(d_losses, g_losses, save_path):
    """Plot and save training loss curves"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(d_losses, label='Discriminator Loss', color='blue')
    plt.plot(g_losses, label='Generator Loss', color='red')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Moving average for smoother visualization
    window = max(1, len(d_losses) // 20)
    d_smooth = np.convolve(d_losses, np.ones(window)/window, mode='valid')
    g_smooth = np.convolve(g_losses, np.ones(window)/window, mode='valid')
    
    plt.plot(d_smooth, label='D Loss (Smoothed)', color='blue')
    plt.plot(g_smooth, label='G Loss (Smoothed)', color='red')
    plt.title('Smoothed Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def validate_gan(generator, discriminator, real_data_sample, noise_dim, device):
    """Validate GAN performance during training"""
    generator.eval()
    discriminator.eval()
    
    with torch.no_grad():
        # Generate fake data
        noise = torch.randn(real_data_sample.size(0), noise_dim).to(device)
        fake_data = generator(noise)
        
        # Get discriminator scores
        real_scores = discriminator(real_data_sample.to(device))
        fake_scores = discriminator(fake_data)
        
        # Calculate metrics
        real_acc = (real_scores > 0.5).float().mean().item()
        fake_acc = (fake_scores < 0.5).float().mean().item()
        
    generator.train()
    discriminator.train()
    
    return real_acc, fake_acc


def train_gan(csv_path, batch_size=32, num_epochs=100, lr=0.0002, 
              categorical_cols=None, save_interval=20, validate_interval=10,
              device=None):
    """
    
    
    Enhanced GAN training with monitoring and validation
    
    Aurg:
    - csv_path: Path to the CSV file containing the dataset
    - batch_size: Size of each training batch
    - num_epochs: Number of training epochs 
    - lr: Learning rate for the optimizers
    - categorical_cols: List of categorical columns to encode (if any)
    - save_interval: Interval to save model checkpoints
    - validate_interval: Interval to validate GAN performance
    - device: Device to run the training on (default is CPU or CUDA if available)
    Returns:
    - generator: Trained generator model
    - discriminator: Trained discriminator model
    - d_losses: List of discriminator losses per epoch
    - g_losses: List of generator losses per epoch
    
    
    """
    
    # Setup
    logger = setup_logging()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Infer dimensions dynamically
    output_dim, noise_dim, dataset = infer_dims(csv_path, categorical_cols)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    logger.info(f"Dataset loaded: {len(dataset)} samples, {output_dim} features")
    logger.info(f"Noise dimension: {noise_dim}")
    
    # Initialize GAN components
    generator = ThreatGenerator(noise_dim=noise_dim, output_dim=output_dim).to(device)
    discriminator = ThreatDiscriminator(input_dim=output_dim).to(device)
    
    # Optimizers with different learning rates (common practice)
    optim_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_D = optim.Adam(discriminator.parameters(), lr=lr*0.5, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()  # More stable than BCELoss
    
    # Training monitoring
    d_losses = []
    g_losses = []
    
    # Create model directory
    model_dir = f"models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save configuration
    config = {
        'csv_path': csv_path,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr': lr,
        'noise_dim': noise_dim,
        'output_dim': output_dim,
        'categorical_cols': categorical_cols,
        'device': str(device)
    }
    save_training_config(config, model_dir)
    
    # Get a sample for validation
    real_sample = next(iter(data_loader))
    
    logger.info("Starting GAN training...")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        batch_count = 0
        
        for real_data in data_loader:
            current_batch_size = real_data.size(0)
            real_data = real_data.to(device)
            
            # Train Discriminator
            optim_D.zero_grad()
            
            # Real data - use label smoothing for better training
            real_labels = torch.ones((current_batch_size, 1)).to(device) * 0.9
            fake_labels = torch.zeros((current_batch_size, 1)).to(device)
            
            # Real data loss
            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake data loss
            noise = torch.randn(current_batch_size, noise_dim).to(device)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            # Backprop Discriminator
            d_loss = (d_loss_real + d_loss_fake) * 0.5
            d_loss.backward()
            optim_D.step()
            
            # Train Generator (less frequently to balance training)
            if batch_count % 1 == 0:  # Train generator every batch
                optim_G.zero_grad()
                fake_output = discriminator(fake_data)
                g_loss = criterion(fake_output, real_labels)  # Fool discriminator
                
                # Backprop Generator
                g_loss.backward()
                optim_G.step()
            
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            batch_count += 1
        
        # Record epoch losses
        avg_d_loss = epoch_d_loss / batch_count
        avg_g_loss = epoch_g_loss / batch_count
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        
        # Validation
        if (epoch + 1) % validate_interval == 0:
            real_acc, fake_acc = validate_gan(generator, discriminator, real_sample, noise_dim, device)
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}] | "
                       f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f} | "
                       f"Real Acc: {real_acc:.3f} | Fake Acc: {fake_acc:.3f}")
        else:
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}] | "
                       f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")
        
        # Save intermediate models
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optim_G': optim_G.state_dict(),
                'optim_D': optim_D.state_dict(),
                'epoch': epoch,
                'config': config
            }, os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final models
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'dataset_scaler': dataset.scaler,
        'feature_names': dataset.feature_names,
        'config': config,
        'training_losses': {'d_losses': d_losses, 'g_losses': g_losses}
    }, os.path.join(model_dir, 'final_model.pth'))
    
    # Plot and save loss curves
    plot_losses(d_losses, g_losses, os.path.join(model_dir, 'training_losses.png'))
    
    logger.info(f"Training complete! Models saved in {model_dir}")
    
    return generator, discriminator, d_losses, g_losses


def generate_threat_samples(generator_path, n_samples=1000, output_csv=None):
    """
    Generate synthetic threat samples using trained generator
    """
    # Load model
    checkpoint = torch.load(generator_path, map_location='cpu')
    config = checkpoint['config']
    
    # Initialize generator
    generator = ThreatGenerator(noise_dim=config['noise_dim'], 
                              output_dim=config['output_dim'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Generate samples
    with torch.no_grad():
        noise = torch.randn(n_samples, config['noise_dim'])
        fake_data = generator(noise).numpy()
    
    # Inverse transform if scaler available
    if 'dataset_scaler' in checkpoint and checkpoint['dataset_scaler'] is not None:
        fake_data = checkpoint['dataset_scaler'].inverse_transform(fake_data)
    
    # Create DataFrame
    feature_names = checkpoint.get('feature_names', 
                                  [f'feature_{i}' for i in range(config['output_dim'])])
    df = pd.DataFrame(fake_data, columns=feature_names)
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Generated {n_samples} synthetic threat samples saved to {output_csv}")
    
    return df


if __name__ == "__main__":
    # Configuration
    csv_path = "C:\\D drive\\ACT_Bench\\data\\train_data.csv"
    batch_size = 32
    num_epochs = 100
    lr = 0.0002
    
    # Specify categorical columns if any (e.g., protocol types, attack types)
    categorical_cols = None  # ['protocol_type', 'service', 'flag']  # Adjust based on your data
    
    # Train GAN
    generator, discriminator, d_losses, g_losses = train_gan(
        csv_path=csv_path,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        categorical_cols=categorical_cols,
        save_interval=25,
        validate_interval=10
    )
    
    # Generate some synthetic samples
    # generate_threat_samples('models_*/final_model.pth', n_samples=500, 
    #                        output_csv='synthetic_threats.csv')