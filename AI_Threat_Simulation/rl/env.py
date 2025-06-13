import gym
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Any
from collections import deque

class ThreatSimulationEnv(gym.Env):
    def __init__(self, generator, ids_model=None, noise_dim=None, output_dim=None, 
                 max_episode_steps=100, detection_threshold=0.5):
        super(ThreatSimulationEnv, self).__init__()
        
        self.generator = generator
        self.ids_model = ids_model  # IDS/IPS model for realistic feedback
        
        # Dynamically assign dimensions
        self.noise_dim = noise_dim if noise_dim is not None else getattr(generator, 'noise_dim', 100)
        self.output_dim = output_dim if output_dim is not None else getattr(generator, 'output_dim', 784)
        
        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.detection_threshold = detection_threshold
        self.current_step = 0
        
        # State tracking for adaptive behavior
        self.attack_history = deque(maxlen=10)  # Track recent attack patterns
        self.detection_history = deque(maxlen=10)  # Track detection results
        self.success_rate = 0.0
        
        # Define spaces
        # Observation includes: current attack pattern + context (detection history, success rate)
        obs_dim = self.output_dim + 11  # +10 for detection history, +1 for success rate
        self.observation_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(obs_dim,))
        
        # Actions: different attack strategies and evasion techniques
        self.action_space = gym.spaces.MultiDiscrete([
            4,  # Attack intensity: [Low, Medium, High, Burst]
            5,  # Attack type: [DDoS, Data Exfiltration, Ransomware, Zero-day, Reconnaissance]
            3,  # Evasion technique: [None, Obfuscation, Fragmentation]
            4   # Timing: [Immediate, Delayed, Periodic, Random]
        ])
        
        # Attack type mappings for realistic simulation
        self.attack_types = {
            0: "ddos",
            1: "exfiltration", 
            2: "ransomware",
            3: "zero_day",
            4: "reconnaissance"
        }
        
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.attack_history.clear()
        self.detection_history.clear()
        self.success_rate = 0.0
        
        # Generate initial attack pattern
        self.noise = torch.randn(1, self.noise_dim)
        self.current_attack = self._generate_attack_pattern()
        
        return self._get_observation()

    def _generate_attack_pattern(self, action=None) -> np.ndarray:
        """Generate attack pattern using GAN, optionally modified by action"""
        if action is not None:
            # Modify noise based on action to create different attack patterns
            intensity, attack_type, evasion, timing = action
            
            # Intensity modification
            intensity_scale = [0.5, 1.0, 1.5, 2.0][intensity]
            modified_noise = self.noise * intensity_scale
            
            # Attack type modification (different noise patterns for different attacks)
            type_offset = torch.randn_like(self.noise) * 0.3 * attack_type
            modified_noise += type_offset
            
            # Evasion technique modification
            if evasion == 1:  # Obfuscation
                modified_noise += torch.randn_like(self.noise) * 0.2
            elif evasion == 2:  # Fragmentation
                modified_noise = modified_noise * (1 + 0.1 * torch.randn_like(modified_noise))
            
            self.noise = modified_noise
        
        # Generate attack pattern using GAN
        with torch.no_grad():
            attack_pattern = self.generator(self.noise).detach().numpy().flatten()
        
        # Ensure correct dimensions
        if attack_pattern.shape[0] != self.output_dim:
            # Resize if necessary
            if attack_pattern.shape[0] > self.output_dim:
                attack_pattern = attack_pattern[:self.output_dim]
            else:
                # Pad with zeros if too small
                padding = np.zeros(self.output_dim - attack_pattern.shape[0])
                attack_pattern = np.concatenate([attack_pattern, padding])
        
        return attack_pattern

    def _get_observation(self) -> np.ndarray:
        """Get current observation including attack pattern and context"""
        # Current attack pattern
        obs = self.current_attack.copy()
        
        # Detection history (last 10 detections, padded with -1 if less than 10)
        detection_hist = list(self.detection_history) + [-1] * (10 - len(self.detection_history))
        
        # Success rate
        success_rate = [self.success_rate]
        
        # Combine all observations
        full_obs = np.concatenate([obs, detection_hist, success_rate])
        
        return full_obs.astype(np.float32)

    def _calculate_detection_probability(self, attack_pattern: np.ndarray, action: Tuple) -> float:
        """Calculate detection probability based on attack pattern and IDS capabilities"""
        if self.ids_model is not None:
            # Use actual IDS model for detection
            with torch.no_grad():
                detection_logits = self.ids_model(torch.tensor(attack_pattern).unsqueeze(0))
                detection_prob = torch.sigmoid(detection_logits).item()
        else:
            # Simulate IDS detection based on attack characteristics
            intensity, attack_type, evasion, timing = action if action else (1, 0, 0, 0)
            
            # Base detection probability based on attack intensity
            base_prob = 0.3 + (intensity * 0.2)
            
            # Attack type affects detection (some attacks are harder to detect)
            type_modifiers = [0.8, 0.6, 0.7, 0.4, 0.9]  # DDoS, Exfil, Ransom, Zero-day, Recon
            base_prob *= type_modifiers[attack_type]
            
            # Evasion techniques reduce detection probability
            evasion_modifiers = [1.0, 0.7, 0.8]  # None, Obfuscation, Fragmentation
            base_prob *= evasion_modifiers[evasion]
            
            # Pattern analysis - repeated similar attacks are more likely to be detected
            if len(self.attack_history) > 0:
                similarity = np.mean([np.corrcoef(attack_pattern, prev_attack)[0,1] 
                                    for prev_attack in self.attack_history 
                                    if not np.isnan(np.corrcoef(attack_pattern, prev_attack)[0,1])])
                if not np.isnan(similarity):
                    base_prob += similarity * 0.3  # Increase detection for similar patterns
            
            detection_prob = np.clip(base_prob, 0.0, 1.0)
        
        return detection_prob

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Convert action to tuple
        action_tuple = tuple(action) if isinstance(action, np.ndarray) else action
        
        # Generate new attack pattern based on action
        self.current_attack = self._generate_attack_pattern(action_tuple)
        
        # Calculate detection probability
        detection_prob = self._calculate_detection_probability(self.current_attack, action_tuple)
        detected = np.random.rand() < detection_prob
        
        # Store in history
        self.attack_history.append(self.current_attack.copy())
        self.detection_history.append(1.0 if detected else 0.0)
        
        # Update success rate
        self.success_rate = 1.0 - np.mean(self.detection_history)
        
        # Calculate reward
        reward = self._calculate_reward(detected, action_tuple, detection_prob)
        
        # Check if episode is done
        done = (self.current_step >= self.max_episode_steps) or (self.success_rate < 0.1)
        
        # Additional info for debugging/analysis
        info = {
            'detected': detected,
            'detection_probability': detection_prob,
            'success_rate': self.success_rate,
            'attack_type': self.attack_types.get(action_tuple[1], 'unknown'),
            'step': self.current_step
        }
        
        return self._get_observation(), reward, done, info

    def _calculate_reward(self, detected: bool, action: Tuple, detection_prob: float) -> float:
        """Calculate reward based on attack success and strategic considerations"""
        intensity, attack_type, evasion, timing = action
        
        # Base reward for successful (undetected) attack
        if not detected:
            base_reward = 1.0
            
            # Bonus for high-value attacks that succeeded
            if attack_type in [1, 2, 3]:  # Exfiltration, Ransomware, Zero-day
                base_reward += 0.5
            
            # Bonus for evading detection with high detection probability
            if detection_prob > 0.7:
                base_reward += 0.3
                
        else:
            # Penalty for being detected
            base_reward = -1.0
            
            # Less penalty if detection was very likely (hard to avoid)
            if detection_prob > 0.9:
                base_reward += 0.3
        
        # Efficiency bonus - prefer successful attacks with lower intensity
        if not detected and intensity < 2:
            base_reward += 0.2
        
        # Diversity bonus - reward for trying different attack types
        if len(self.attack_history) > 1:
            recent_types = [self.attack_types.get(action[1], 'unknown') 
                          for action in [action]]  # Could track action history too
            # This is simplified - you could track action history for better diversity rewards
        
        # Stealth bonus - consistent success over time
        if self.success_rate > 0.8 and len(self.detection_history) > 5:
            base_reward += 0.4
        
        return base_reward

    def render(self, mode='human'):
        """Render current state (optional, for debugging)"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Success Rate: {self.success_rate:.2f}")
            print(f"Recent Detections: {list(self.detection_history)[-5:]}")
            print(f"Attack Pattern Stats: min={self.current_attack.min():.3f}, "
                  f"max={self.current_attack.max():.3f}, mean={self.current_attack.mean():.3f}")
            print("-" * 50)

# Example usage and training setup
class ThreatSimulationTrainer:
    """Helper class for training RL agents in the threat simulation environment"""
    
    def __init__(self, generator, ids_model=None):
        self.env = ThreatSimulationEnv(generator, ids_model)
        self.training_stats = {
            'episodes': 0,
            'total_rewards': [],
            'success_rates': [],
            'detection_rates': []
        }
    
    def collect_episode(self, agent, max_steps=100):
        """Collect one episode of experience"""
        obs = self.env.reset()
        episode_reward = 0
        episode_detections = 0
        episode_steps = 0
        
        experiences = []
        
        for step in range(max_steps):
            # Agent selects action
            action = agent.select_action(obs)
            
            # Environment step
            next_obs, reward, done, info = self.env.step(action)
            
            # Store experience
            experiences.append((obs, action, reward, next_obs, done))
            
            # Update metrics
            episode_reward += reward
            episode_detections += int(info['detected'])
            episode_steps += 1
            
            obs = next_obs
            
            if done:
                break
        
        # Update training stats
        self.training_stats['episodes'] += 1
        self.training_stats['total_rewards'].append(episode_reward)
        self.training_stats['success_rates'].append(info.get('success_rate', 0))
        self.training_stats['detection_rates'].append(episode_detections / episode_steps)
        
        return experiences, episode_reward, info
    
    def get_training_stats(self):
        """Get current training statistics"""
        if not self.training_stats['total_rewards']:
            return {}
        
        return {
            'episodes': self.training_stats['episodes'],
            'avg_reward': np.mean(self.training_stats['total_rewards'][-100:]),
            'avg_success_rate': np.mean(self.training_stats['success_rates'][-100:]),
            'avg_detection_rate': np.mean(self.training_stats['detection_rates'][-100:])
        }