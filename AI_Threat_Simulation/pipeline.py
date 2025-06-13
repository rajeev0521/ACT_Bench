import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLAgent:
    """RL Agent with policy gradient implementation"""
    
    def __init__(self, state_dim, action_space=3, learning_rate=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Policy and value networks
        self.policy_network = self._build_policy_network()
        self.value_network = self._build_value_network()
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.experience_buffer = []
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _build_policy_network(self):
        """Build policy network for action selection"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space),
            nn.Softmax(dim=-1)
        )
    
    def _build_value_network(self):
        """Build value network for state value estimation"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def choose_action(self, state):
        """Choose action using policy network"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy_network(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            
        return action.item(), dist.log_prob(action).item()
    
    def store_experience(self, state, action, reward, next_state, log_prob, done):
        """Store experience for batch learning"""
        self.experience_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'log_prob': log_prob,
            'done': done
        })
    
    def learn(self):
        """Learn from stored experiences using Actor-Critic"""
        if len(self.experience_buffer) < 32:  # Minimum batch size
            return
        
        # Convert experiences to tensors
        states = torch.FloatTensor([exp['state'] for exp in self.experience_buffer])
        actions = torch.LongTensor([exp['action'] for exp in self.experience_buffer])
        rewards = torch.FloatTensor([exp['reward'] for exp in self.experience_buffer])
        next_states = torch.FloatTensor([exp['next_state'] for exp in self.experience_buffer])
        log_probs = torch.FloatTensor([exp['log_prob'] for exp in self.experience_buffer])
        dones = torch.BoolTensor([exp['done'] for exp in self.experience_buffer])
        
        # Calculate returns
        returns = self._calculate_returns(rewards, dones)
        
        # Get state values
        state_values = self.value_network(states).squeeze()
        next_state_values = self.value_network(next_states).squeeze()
        
        # Calculate advantages
        advantages = returns - state_values.detach()
        
        # Policy loss (Actor)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss (Critic)
        value_loss = nn.MSELoss()(state_values, returns)
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Clear experience buffer
        self.experience_buffer.clear()
        
        return policy_loss.item(), value_loss.item()
    
    def _calculate_returns(self, rewards, dones):
        """Calculate discounted returns"""
        returns = []
        running_return = 0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                running_return = 0
            running_return = rewards[i] + self.gamma * running_return
            returns.insert(0, running_return)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
        return returns

class GANManager:
    """Manages GAN training and attack generation"""
    
    def __init__(self, generator, discriminator, noise_dim, output_dim, device='cpu'):
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.device = device
        
        # Optimizers
        self.g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training metrics
        self.g_losses = []
        self.d_losses = []
        
    def generate_attack(self, batch_size=1):
        """Generate synthetic attack using GAN"""
        with torch.no_grad():
            noise = torch.randn(batch_size, self.noise_dim, device=self.device)
            attack = self.generator(noise)
        return attack.cpu().numpy()
    
    def retrain_gan(self, rl_feedback=None, num_epochs=10, batch_size=32):
        """Retrain GAN with optional RL feedback"""
        epoch_g_losses = []
        epoch_d_losses = []
        
        for epoch in range(num_epochs):
            # Generate real-like data (simulated)
            real_data = self._generate_real_data(batch_size)
            
            # Train Discriminator
            d_loss = self._train_discriminator_step(real_data, batch_size)
            
            # Train Generator
            g_loss = self._train_generator_step(batch_size, rl_feedback)
            
            epoch_g_losses.append(g_loss)
            epoch_d_losses.append(d_loss)
        
        self.g_losses.extend(epoch_g_losses)
        self.d_losses.extend(epoch_d_losses)
        
        return np.mean(epoch_g_losses), np.mean(epoch_d_losses)
    
    def _generate_real_data(self, batch_size):
        """Generate realistic threat data for training"""
        # Simulate different types of real threat patterns
        real_data = []
        for _ in range(batch_size):
            # Create realistic threat signatures
            threat_type = np.random.choice(['ddos', 'exfiltration', 'ransomware', 'recon'])
            
            if threat_type == 'ddos':
                pattern = np.random.exponential(2.0, self.output_dim)
            elif threat_type == 'exfiltration':
                pattern = np.random.lognormal(0, 1, self.output_dim)
            elif threat_type == 'ransomware':
                pattern = np.random.weibull(1.5, self.output_dim)
            else:  # reconnaissance
                pattern = np.random.beta(2, 5, self.output_dim)
            
            # Normalize
            pattern = (pattern - pattern.mean()) / (pattern.std() + 1e-8)
            real_data.append(pattern)
        
        return torch.FloatTensor(real_data).to(self.device)
    
    def _train_discriminator_step(self, real_data, batch_size):
        """Single discriminator training step"""
        self.d_optimizer.zero_grad()
        
        # Real data
        real_labels = torch.ones(batch_size, 1, device=self.device)
        real_output = self.discriminator(real_data)
        real_loss = self.criterion(real_output, real_labels)
        
        # Fake data
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_data = self.generator(noise).detach()
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        fake_output = self.discriminator(fake_data)
        fake_loss = self.criterion(fake_output, fake_labels)
        
        # Total loss
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
    
    def _train_generator_step(self, batch_size, rl_feedback=None):
        """Single generator training step with optional RL feedback"""
        self.g_optimizer.zero_grad()
        
        # Generate fake data
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_data = self.generator(noise)
        fake_output = self.discriminator(fake_data)
        
        # Adversarial loss
        real_labels = torch.ones(batch_size, 1, device=self.device)
        g_loss = self.criterion(fake_output, real_labels)
        
        # Add RL feedback if available
        if rl_feedback is not None:
            # Incorporate RL success rate into generator training
            success_rate = rl_feedback.get('success_rate', 0.5)
            if success_rate > 0.8:  # RL too successful, make attacks more challenging
                diversity_loss = -torch.var(fake_data, dim=1).mean()
                g_loss += 0.1 * diversity_loss
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()

class EnhancedContinuousTrainingPipeline:
    """Enhanced training pipeline with proper integration"""
    
    def __init__(self, generator, discriminator, attack_env, noise_dim, output_dim, 
                 num_epochs=100, device='cpu'):
        self.device = device
        self.num_epochs = num_epochs
        self.attack_env = attack_env
        
        # Initialize components
        self.gan_manager = GANManager(generator, discriminator, noise_dim, output_dim, device)
        
        # Get state dimension from environment
        state_dim = attack_env.observation_space.shape[0]
        action_dim = attack_env.action_space.n if hasattr(attack_env.action_space, 'n') else 3
        
        self.rl_agent = RLAgent(state_dim, action_dim)
        
        # Training metrics
        self.training_metrics = {
            'epoch_rewards': [],
            'epoch_detection_rates': [],
            'gan_losses': [],
            'rl_success_rates': []
        }
        
        # Save configuration
        self.config = {
            'noise_dim': noise_dim,
            'output_dim': output_dim,
            'num_epochs': num_epochs,
            'device': str(device)
        }
    
    def run_episode(self, max_steps=200):
        """Run single episode with RL agent"""
        state = self.attack_env.reset()
        episode_reward = 0
        episode_length = 0
        detections = 0
        
        for step in range(max_steps):
            # Choose action
            action, log_prob = self.rl_agent.choose_action(state)
            
            # Take step in environment
            next_state, reward, done, info = self.attack_env.step(action)
            
            # Store experience
            self.rl_agent.store_experience(state, action, reward, next_state, log_prob, done)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            if info.get('detected', False):
                detections += 1
            
            state = next_state
            
            if done:
                break
        
        # Learn from episode
        losses = self.rl_agent.learn()
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'detection_rate': detections / episode_length,
            'losses': losses
        }
    
    def continuous_training(self):
        """Main training loop"""
        logger.info("Starting continuous training pipeline...")
        logger.info(f"Configuration: {self.config}")
        
        try:
            for epoch in range(self.num_epochs):
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
                
                # Run multiple episodes for RL training
                epoch_rewards = []
                epoch_detection_rates = []
                
                for episode in range(5):  # Multiple episodes per epoch
                    episode_results = self.run_episode()
                    epoch_rewards.append(episode_results['episode_reward'])
                    epoch_detection_rates.append(episode_results['detection_rate'])
                
                # Calculate RL performance metrics
                avg_reward = np.mean(epoch_rewards)
                avg_detection_rate = np.mean(epoch_detection_rates)
                success_rate = 1.0 - avg_detection_rate
                
                # Prepare RL feedback for GAN
                rl_feedback = {
                    'success_rate': success_rate,
                    'avg_reward': avg_reward,
                    'detection_rate': avg_detection_rate
                }
                
                # Retrain GAN with RL feedback
                if epoch % 5 == 0:  # Retrain GAN every 5 epochs
                    logger.info("Retraining GAN with RL feedback...")
                    g_loss, d_loss = self.gan_manager.retrain_gan(rl_feedback, num_epochs=10)
                    logger.info(f"GAN Losses - G: {g_loss:.4f}, D: {d_loss:.4f}")
                
                # Store metrics
                self.training_metrics['epoch_rewards'].append(avg_reward)
                self.training_metrics['epoch_detection_rates'].append(avg_detection_rate)
                self.training_metrics['rl_success_rates'].append(success_rate)
                
                # Log progress
                logger.info(f"Epoch {epoch + 1} Results:")
                logger.info(f"  Avg Reward: {avg_reward:.3f}")
                logger.info(f"  Detection Rate: {avg_detection_rate:.3f}")
                logger.info(f"  Success Rate: {success_rate:.3f}")
                
                # Save models periodically
                if epoch % 20 == 0:
                    self.save_models(f"checkpoint_epoch_{epoch}")
                
                # Optional: Early stopping based on performance
                if len(self.training_metrics['rl_success_rates']) > 10:
                    recent_success = np.mean(self.training_metrics['rl_success_rates'][-10:])
                    if recent_success > 0.95:
                        logger.info("Early stopping: RL agent too successful")
                        break
                
                time.sleep(1)  # Brief pause between epochs
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
        finally:
            # Save final models and results
            self.save_models("final")
            self.save_training_results()
            self.plot_training_metrics()
            logger.info("Training pipeline completed")
    
    def save_models(self, suffix=""):
        """Save all models"""
        os.makedirs("models", exist_ok=True)
        
        # Save GAN models
        torch.save(self.gan_manager.generator.state_dict(), f"models/generator_{suffix}.pth")
        torch.save(self.gan_manager.discriminator.state_dict(), f"models/discriminator_{suffix}.pth")
        
        # Save RL agent
        torch.save({
            'policy_network': self.rl_agent.policy_network.state_dict(),
            'value_network': self.rl_agent.value_network.state_dict(),
            'config': self.config
        }, f"models/rl_agent_{suffix}.pth")
        
        logger.info(f"Models saved with suffix: {suffix}")
    
    def save_training_results(self):
        """Save training metrics"""
        os.makedirs("results", exist_ok=True)
        
        with open("results/training_metrics.json", "w") as f:
            json.dump(self.training_metrics, f, indent=2)
        
        logger.info("Training results saved")
    
    def plot_training_metrics(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # RL Rewards
        axes[0, 0].plot(self.training_metrics['epoch_rewards'])
        axes[0, 0].set_title('RL Agent Rewards per Epoch')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].grid(True)
        
        # Detection Rates
        axes[0, 1].plot(self.training_metrics['epoch_detection_rates'])
        axes[0, 1].set_title('Detection Rates per Epoch')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Detection Rate')
        axes[0, 1].grid(True)
        
        # Success Rates
        axes[1, 0].plot(self.training_metrics['rl_success_rates'])
        axes[1, 0].set_title('RL Success Rates per Epoch')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].grid(True)
        
        # GAN Losses
        if self.gan_manager.g_losses:
            axes[1, 1].plot(self.gan_manager.g_losses, label='Generator')
            axes[1, 1].plot(self.gan_manager.d_losses, label='Discriminator')
            axes[1, 1].set_title('GAN Training Losses')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Training plots saved")

# Utility functions
def get_dynamic_dims(complexity_level='medium'):
    """Calculate dimensions based on complexity level"""
    if complexity_level == 'low':
        noise_dim, output_dim = 50, 50
    elif complexity_level == 'medium':
        noise_dim, output_dim = 100, 100
    elif complexity_level == 'high':
        noise_dim, output_dim = 200, 200
    else:
        noise_dim, output_dim = 100, 100
    
    return noise_dim, output_dim

def create_mock_environment(generator, noise_dim, output_dim):
    """Create a mock environment for testing"""
    class MockEnv:
        def __init__(self, generator, noise_dim, output_dim):
            self.generator = generator
            self.noise_dim = noise_dim
            self.output_dim = output_dim
            self.observation_space = type('obj', (object,), {'shape': (output_dim + 1,)})()
            self.action_space = type('obj', (object,), {'n': 3})()
            self.current_state = None
        
        def reset(self):
            noise = torch.randn(1, self.noise_dim)
            self.current_state = self.generator(noise).detach().numpy().flatten()
            # Add success rate to state
            return np.concatenate([self.current_state, [0.5]])
        
        def step(self, action):
            reward = np.random.uniform(-1, 1)
            done = np.random.rand() > 0.9
            detected = np.random.rand() > 0.6
            
            # Generate next state
            noise = torch.randn(1, self.noise_dim)
            self.current_state = self.generator(noise).detach().numpy().flatten()
            next_state = np.concatenate([self.current_state, [0.5]])
            
            info = {'detected': detected}
            return next_state, reward, done, info
    
    return MockEnv(generator, noise_dim, output_dim)

# Main execution
if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get dynamic dimensions
    noise_dim, output_dim = get_dynamic_dims('medium')
    logger.info(f"Using dimensions - Noise: {noise_dim}, Output: {output_dim}")
    
    # Import or create your GAN models
    try:
        from gan import ThreatGenerator, ThreatDiscriminator
        generator = ThreatGenerator(noise_dim=noise_dim, output_dim=output_dim).to(device)
        discriminator = ThreatDiscriminator(input_dim=output_dim).to(device)
        logger.info("Loaded GAN models successfully")
    except ImportError:
        logger.warning("GAN models not found, using mock models")
        # Create simple mock models for testing
        generator = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        ).to(device)
        
        discriminator = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)
    
    # Create environment
    try:
        from env import ThreatMitigationEnv
        attack_env = ThreatMitigationEnv(generator=generator, noise_dim=noise_dim, output_dim=output_dim)
        logger.info("Loaded custom environment successfully")
    except ImportError:
        logger.warning("Custom environment not found, using mock environment")
        attack_env = create_mock_environment(generator, noise_dim, output_dim)
    
    # Create and run pipeline
    pipeline = EnhancedContinuousTrainingPipeline(
        generator=generator,
        discriminator=discriminator,
        attack_env=attack_env,
        noise_dim=noise_dim,
        output_dim=output_dim,
        num_epochs=100,
        device=device
    )
    
    # Start training
    pipeline.continuous_training()