import gym
import numpy as np
import torch

class ThreatMitigationEnv(gym.Env):
    def __init__(self, generator, noise_dim=None, output_dim=None):
        super(ThreatMitigationEnv, self).__init__()
        self.generator = generator
        
        # Dynamically assign dimensions if not passed
        self.noise_dim = noise_dim if noise_dim is not None else generator.noise_dim
        self.output_dim = output_dim if output_dim is not None else generator.output_dim
        
        self.noise = torch.randn(1, self.noise_dim)

        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.output_dim,))
        self.action_space = gym.spaces.Discrete(3)  # Actions: Mitigate, Ignore, Log

        self.current_state = None

    def reset(self):
        # Generate a new state using the generator
        self.noise = torch.randn(1, self.noise_dim)
        self.current_state = self.generator(self.noise).detach().numpy().flatten()
        
        if self.current_state.shape[0] != self.output_dim:
            raise ValueError(f"Generated state has shape {self.current_state.shape}, expected shape ({self.output_dim},)")
    
        return self.current_state

    def step(self, action):
        # Enhanced reward shaping: reward for mitigating specific patterns
        if action == 0:  # Mitigate
            reward = 1.0 if np.mean(self.current_state) > 0 else -1.0
        elif action == 1:  # Ignore
            reward = -0.5  # Penalize ignoring threats
        else:  # Log
            reward = 0.2  # Small neutral reward for logging

        # Feedback: Adjust noise based on action
        if action == 0:
            self.noise += torch.randn_like(self.noise) * 0.1  # Reward mitigation with slight noise variation
        elif action == 1:
            self.noise -= torch.randn_like(self.noise) * 0.1  # Penalize ignoring with reduced noise

        # Generate next state
        self.current_state = self.generator(self.noise).detach().numpy().flatten()
        if self.current_state.shape[0] != self.output_dim:
            raise ValueError(f"Generated state has shape {self.current_state.shape}, expected shape ({self.output_dim},)")
        
        done = np.random.rand() > 0.95  # Random chance to terminate episode
        return self.current_state, reward, done, {}
