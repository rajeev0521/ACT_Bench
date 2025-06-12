import torch
from torch import nn, optim
from stable_baselines3 import PPO
from .env import ThreatMitigationEnv
import os
import sys

sys.path.append('C:\\Users\\rajee\\OneDrive\\Desktop\\ACT_Bench Tool\\geekbuddies\\AI_Threat_Simulation')
from gan import *  # Assuming generator and discriminator are in gan.py

# Setup: Load GAN components
output_dim = 10
noise_dim = max(10, output_dim // 2) 
generator = ThreatGenerator(noise_dim=noise_dim, output_dim=output_dim)
discriminator = ThreatDiscriminator(input_dim=generator.output_dim)  # Use the dynamically assigned output_dim

# Load pre-trained models
generator_path = "C:\\Users\\rajee\\OneDrive\\Desktop\\ACT_Bench Tool\\models\\generator.pth"
discriminator_path = "C:\\Users\\rajee\\OneDrive\\Desktop\\ACT_Bench Tool\\models\\discriminator.pth"

if os.path.exists(generator_path):
    generator.load_state_dict(torch.load(generator_path))
else:
    print(f"{generator_path} not found. Initializing and saving a new generator.")
    torch.save(generator.state_dict(), generator_path)

if os.path.exists(discriminator_path):
    discriminator.load_state_dict(torch.load(discriminator_path))
else:
    print(f"{discriminator_path} not found. Initializing and saving a new discriminator.")
    torch.save(discriminator.state_dict(), discriminator_path)

# Training configuration for GAN
gan_optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
gan_optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
gan_criterion = nn.BCELoss()

# RL environment
env = ThreatMitigationEnv(generator=generator, noise_dim=generator.noise_dim, output_dim=generator.output_dim)
model = PPO("MlpPolicy", env, verbose=1)

def retrain_gan(generator, discriminator, gan_optimizer_G, gan_optimizer_D, gan_criterion, num_steps=100):
    """Retrain GAN based on current noise and discriminator feedback."""
    for _ in range(num_steps):
        # Generate fake data
        noise = torch.randn(32, generator.noise_dim)
        fake_data = generator(noise)

        # Create labels
        real_labels = torch.ones(32, 1)
        fake_labels = torch.zeros(32, 1)

        # Train Discriminator
        gan_optimizer_D.zero_grad()
        real_data = torch.randn(32, generator.output_dim)  # Example: Real threat data
        real_output = discriminator(real_data)
        fake_output = discriminator(fake_data.detach())
        d_loss = gan_criterion(real_output, real_labels) + gan_criterion(fake_output, fake_labels)
        d_loss.backward()
        gan_optimizer_D.step()

        # Train Generator
        gan_optimizer_G.zero_grad()
        fake_output = discriminator(fake_data)
        g_loss = gan_criterion(fake_output, real_labels)
        g_loss.backward()
        gan_optimizer_G.step()

# Train RL agent with GAN retraining
for step in range(10):  # Outer loop for RL-GAN co-evolution
    print(f"RL Training Step: {step + 1}")
    model.learn(total_timesteps=1000)

    # Retrain GAN dynamically
    print(f"Retraining GAN...")
    retrain_gan(generator, discriminator, gan_optimizer_G, gan_optimizer_D, gan_criterion)

# Save RL agent and updated GAN
model.save("rl_agent")
torch.save(generator.state_dict(), "updated_generator.pth")
torch.save(discriminator.state_dict(), "updated_discriminator.pth")
