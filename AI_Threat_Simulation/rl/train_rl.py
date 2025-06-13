import torch
from torch import nn, optim
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import sys
import wandb  # for experiment tracking

# Add your project path
sys.path.append('C:\\D drive\\ACT_Bench\\AI_Threat_Simulation')
from gan import *  # Your GAN implementations
from env import ThreatSimulationEnv  # Use the enhanced environment

class AdversarialTrainingCallback(BaseCallback):
    """Custom callback for coordinated GAN-RL training"""
    
    def __init__(self, generator, discriminator, gan_trainer, retrain_frequency=1000, verbose=0):
        super(AdversarialTrainingCallback, self).__init__(verbose)
        self.generator = generator
        self.discriminator = discriminator
        self.gan_trainer = gan_trainer
        self.retrain_frequency = retrain_frequency
        self.step_count = 0
        self.training_metrics = {
            'rl_rewards': deque(maxlen=100),
            'gan_losses': deque(maxlen=100),
            'success_rates': deque(maxlen=100)
        }

    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Collect RL metrics
        if len(self.locals.get('rewards', [])) > 0:
            self.training_metrics['rl_rewards'].extend(self.locals['rewards'])
        
        # Retrain GAN periodically
        if self.step_count % self.retrain_frequency == 0:
            print(f"\n=== Retraining GAN at step {self.step_count} ===")
            
            # Get recent RL experiences for GAN training
            rl_success_rate = self._get_recent_success_rate()
            
            # Retrain GAN with adaptive strategy
            gan_losses = self.gan_trainer.adaptive_retrain(
                success_rate=rl_success_rate,
                num_epochs=50
            )
            
            self.training_metrics['gan_losses'].extend(gan_losses)
            self.training_metrics['success_rates'].append(rl_success_rate)
            
            # Log metrics
            if self.verbose > 0:
                avg_reward = np.mean(self.training_metrics['rl_rewards']) if self.training_metrics['rl_rewards'] else 0
                avg_gan_loss = np.mean(gan_losses) if gan_losses else 0
                print(f"Avg RL Reward: {avg_reward:.3f}, GAN Loss: {avg_gan_loss:.3f}, Success Rate: {rl_success_rate:.3f}")
        
        return True
    
    def _get_recent_success_rate(self):
        """Calculate recent success rate from RL training"""
        # This is a simplified version - you might want to get actual success rate from environment
        if self.training_metrics['rl_rewards']:
            # Approximate success rate based on positive rewards
            recent_rewards = list(self.training_metrics['rl_rewards'])[-20:]
            success_rate = np.mean([1 if r > 0 else 0 for r in recent_rewards])
            return success_rate
        return 0.5

class AdversarialGANTrainer:
    """Enhanced GAN trainer with adversarial feedback"""
    
    def __init__(self, generator, discriminator, ids_model=None, device='cpu'):
        self.generator = generator
        self.discriminator = discriminator
        self.ids_model = ids_model
        self.device = device
        
        # Optimizers
        self.optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'g_losses': [],
            'd_losses': [],
            'detection_rates': []
        }
    
    def adaptive_retrain(self, success_rate, num_epochs=50, batch_size=64):
        """Adaptively retrain GAN based on RL performance"""
        losses = []
        
        # Adjust training intensity based on RL success rate
        if success_rate > 0.8:
            # RL is too successful, make discriminator stronger
            d_steps = 3
            g_steps = 1
            lr_multiplier = 1.2
        elif success_rate < 0.3:
            # RL is struggling, make generator stronger
            d_steps = 1
            g_steps = 3
            lr_multiplier = 0.8
        else:
            # Balanced training
            d_steps = 2
            g_steps = 2
            lr_multiplier = 1.0
        
        # Adjust learning rates
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = 0.0001 * lr_multiplier
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = 0.0001 * lr_multiplier
        
        for epoch in range(num_epochs):
            # Generate training data
            real_data = self._generate_real_threat_data(batch_size)
            noise = torch.randn(batch_size, self.generator.noise_dim, device=self.device)
            
            # Train Discriminator
            d_loss_total = 0
            for _ in range(d_steps):
                d_loss = self._train_discriminator_step(real_data, noise)
                d_loss_total += d_loss
            
            # Train Generator  
            g_loss_total = 0
            for _ in range(g_steps):
                g_loss = self._train_generator_step(noise, success_rate)
                g_loss_total += g_loss
            
            avg_d_loss = d_loss_total / d_steps
            avg_g_loss = g_loss_total / g_steps
            losses.append((avg_g_loss, avg_d_loss))
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}")
        
        # Update training history
        epoch_losses = [g_loss for g_loss, _ in losses]
        self.training_history['g_losses'].extend(epoch_losses)
        self.training_history['d_losses'].extend([d_loss for _, d_loss in losses])
        
        return epoch_losses
    
    def _generate_real_threat_data(self, batch_size):
        """Generate or load real threat data for training"""
        # In practice, you would load real network traffic data
        # For now, we'll simulate realistic threat patterns
        
        # Create different types of realistic threat signatures
        threat_types = np.random.choice([0, 1, 2, 3, 4], batch_size)  # Different attack types
        real_data = []
        
        for threat_type in threat_types:
            if threat_type == 0:  # DDoS pattern
                pattern = np.random.exponential(2.0, self.generator.output_dim)
            elif threat_type == 1:  # Data exfiltration pattern
                pattern = np.random.lognormal(0, 1, self.generator.output_dim)
            elif threat_type == 2:  # Ransomware pattern
                pattern = np.random.weibull(1.5, self.generator.output_dim)
            elif threat_type == 3:  # Zero-day pattern
                pattern = np.random.gamma(2, 1, self.generator.output_dim)
            else:  # Reconnaissance pattern
                pattern = np.random.beta(2, 5, self.generator.output_dim)
            
            # Normalize and add noise
            pattern = (pattern - pattern.mean()) / (pattern.std() + 1e-8)
            pattern += np.random.normal(0, 0.1, self.generator.output_dim)
            real_data.append(pattern)
        
        return torch.tensor(real_data, dtype=torch.float32, device=self.device)
    
    def _train_discriminator_step(self, real_data, noise):
        """Single discriminator training step"""
        self.optimizer_D.zero_grad()
        
        batch_size = real_data.size(0)
        
        # Real data
        real_labels = torch.ones(batch_size, 1, device=self.device)
        real_output = self.discriminator(real_data)
        real_loss = self.adversarial_loss(real_output, real_labels)
        
        # Fake data
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        fake_data = self.generator(noise).detach()
        fake_output = self.discriminator(fake_data)
        fake_loss = self.adversarial_loss(fake_output, fake_labels)
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()
        
        return d_loss.item()
    
    def _train_generator_step(self, noise, success_rate):
        """Single generator training step with RL feedback"""
        self.optimizer_G.zero_grad()
        
        batch_size = noise.size(0)
        
        # Generate fake data
        fake_data = self.generator(noise)
        fake_output = self.discriminator(fake_data)
        
        # Basic adversarial loss
        real_labels = torch.ones(batch_size, 1, device=self.device)
        adversarial_loss = self.adversarial_loss(fake_output, real_labels)
        
        # RL-informed loss component
        rl_loss = 0
        if self.ids_model is not None:
            # If we have an IDS model, try to minimize detection
            with torch.no_grad():
                detection_probs = torch.sigmoid(self.ids_model(fake_data))
            # Reward generator for creating hard-to-detect patterns
            rl_loss = detection_probs.mean()
        else:
            # Use success rate to adjust generator training
            # If RL is very successful, make patterns more challenging
            if success_rate > 0.7:
                # Add complexity/diversity loss
                diversity_loss = -torch.var(fake_data, dim=1).mean()
                rl_loss = diversity_loss * 0.1
        
        # Combined loss
        g_loss = adversarial_loss + rl_loss
        g_loss.backward()
        self.optimizer_G.step()
        
        return g_loss.item()

def setup_training_environment(generator, discriminator, ids_model=None):
    """Setup the complete training environment"""
    
    # Create environment
    env = ThreatSimulationEnv(
        generator=generator,
        ids_model=ids_model,
        max_episode_steps=200,
        detection_threshold=0.5
    )
    
    # Wrap in vector environment for stable-baselines3
    vec_env = DummyVecEnv([lambda: env])
    
    # Create PPO model with custom policy network
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        policy_kwargs={
            "net_arch": [dict(pi=[128, 128], vf=[128, 128])]
        }
    )
    
    return env, vec_env, model

def main():
    """Main training loop"""
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters
    output_dim = 100  # Increased for more complex patterns
    noise_dim = 50
    
    # Initialize models
    generator = ThreatGenerator(noise_dim=noise_dim, output_dim=output_dim).to(device)
    discriminator = ThreatDiscriminator(input_dim=output_dim).to(device)
    
    # Load pre-trained models if available
    generator_path = "C:\\Users\\rajee\\OneDrive\\Desktop\\ACT_Bench Tool\\models\\generator.pth"
    discriminator_path = "C:\\Users\\rajee\\OneDrive\\Desktop\\ACT_Bench Tool\\models\\discriminator.pth"
    
    if os.path.exists(generator_path):
        generator.load_state_dict(torch.load(generator_path, map_location=device))
        print("Loaded pre-trained generator")
    
    if os.path.exists(discriminator_path):
        discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
        print("Loaded pre-trained discriminator")
    
    # Setup training components
    gan_trainer = AdversarialGANTrainer(generator, discriminator, device=device)
    env, vec_env, rl_model = setup_training_environment(generator, discriminator)
    
    # Setup callback for coordinated training
    callback = AdversarialTrainingCallback(
        generator=generator,
        discriminator=discriminator,
        gan_trainer=gan_trainer,
        retrain_frequency=2000,  # Retrain GAN every 2000 RL steps
        verbose=1
    )
    
    # Training configuration
    total_rl_timesteps = 100000
    save_frequency = 10000
    
    print("Starting adversarial training...")
    print(f"Total timesteps: {total_rl_timesteps}")
    print(f"GAN retrain frequency: {callback.retrain_frequency}")
    
    try:
        # Main training loop
        rl_model.learn(
            total_timesteps=total_rl_timesteps,
            callback=callback,
            tb_log_name="adversarial_training"
        )
        
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    except Exception as e:
        print(f"Training error: {e}")
        raise
    
    finally:
        # Save final models
        print("Saving final models...")
        rl_model.save("models/final_rl_agent")
        torch.save(generator.state_dict(), "models/final_generator.pth")
        torch.save(discriminator.state_dict(), "models/final_discriminator.pth")
        
        # Plot training metrics
        plot_training_metrics(callback.training_metrics, gan_trainer.training_history)
        
        print("Training session completed!")

def plot_training_metrics(rl_metrics, gan_metrics):
    """Plot training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # RL Rewards
    if rl_metrics['rl_rewards']:
        axes[0, 0].plot(rl_metrics['rl_rewards'])
        axes[0, 0].set_title('RL Agent Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
    
    # GAN Losses
    if gan_metrics['g_losses']:
        axes[0, 1].plot(gan_metrics['g_losses'], label='Generator')
        axes[0, 1].plot(gan_metrics['d_losses'], label='Discriminator')
        axes[0, 1].set_title('GAN Training Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
    
    # Success Rate
    if rl_metrics['success_rates']:
        axes[1, 0].plot(rl_metrics['success_rates'])
        axes[1, 0].set_title('RL Success Rate Over Time')
        axes[1, 0].set_xlabel('GAN Retrain Cycle')
        axes[1, 0].set_ylabel('Success Rate')
    
    # Combined metrics
    if rl_metrics['rl_rewards'] and gan_metrics['g_losses']:
        # Create a correlation plot or other combined visualization
        axes[1, 1].text(0.5, 0.5, 'Training Summary\nCompleted Successfully', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Training Summary')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()