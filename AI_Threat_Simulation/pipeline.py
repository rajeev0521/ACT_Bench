import torch
from gan.generator import ThreatGenerator
from rl.train_rl import PPO, retrain_gan
from rl.env import ThreatMitigationEnv
import numpy as np
import os
import time

class ContinuousTrainingPipeline:
    def __init__(self, gan_model, rl_agent, attack_env, num_epochs=100):
        """
        Initializes the training pipeline.

        :param gan_model: The GAN model used for generating synthetic attacks
        :param rl_agent: The RL agent responsible for adapting to the attacks
        :param attack_env: The environment that simulates interactions between the RL agent and the IDS/IPS
        :param num_epochs: Number of epochs for continuous training
        """
        self.gan_model = gan_model
        self.rl_agent = rl_agent
        self.attack_env = attack_env
        self.num_epochs = num_epochs

    def generate_attack(self):
        """
        Generates a new synthetic attack using the GAN generator.

        :return: A synthetic attack
        """
        attack = self.gan_model.generate_attack()
        return attack

    def train_rl_agent(self, attack):
        """
        Trains the RL agent with the latest attack.

        :param attack: The generated attack to evaluate
        """
        # Reset environment to simulate new attack
        state = self.attack_env.reset(attack)

        # Let the agent take action and learn from the environment
        done = False
        while not done:
            action = self.rl_agent.choose_action(state)
            next_state, reward, done, _ = self.attack_env.step(action)
            self.rl_agent.learn(state, action, reward, next_state)
            state = next_state

    def continuous_training(self):
        """
        Runs the continuous assessment and training loop.

        :return: None
        """
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs} - Generating and assessing attack...")

            # Generate a new attack using GAN
            attack = self.generate_attack()

            # Train RL agent on the generated attack
            self.train_rl_agent(attack)

            # Optionally, add performance metrics or logging here

            # Pause between epochs to simulate time taken for training
            time.sleep(2)

            print(f"Epoch {epoch + 1} complete.\n")


# Dynamically calculate noise_dim and output_dim based on your configuration
def get_dynamic_dims():
    # Example dynamic calculation for noise_dim and output_dim
    # For noise_dim, you can base it on your architecture's complexity or set a default range
    noise_dim = 100  # You can modify this based on your needs, or use a random number generator
    output_dim = 10  # Modify this based on the complexity or dataset

    # You could also use logic to calculate it from the dataset or model architecture:
    # e.g., if you know the dataset's feature size or have specific rules
    return noise_dim, output_dim


class RLAgent:
    def __init__(self, action_space=3, learning_rate=0.01):
        """
        Initializes the RL Agent.
        
        :param action_space: The number of possible actions (e.g., mitigate, ignore, log)
        :param learning_rate: The learning rate for the agent's learning process
        """
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.policy_network = self.build_network()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

    def build_network(self):
        """
        Builds a simple policy network for the RL agent.

        :return: The policy network
        """
        return torch.nn.Sequential(
            torch.nn.Linear(10, 64),  # Assuming the state has 10 features
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.action_space)
        )

    def choose_action(self, state):
        """
        Chooses an action based on the current state using epsilon-greedy approach.
        
        :param state: The current state of the environment
        :return: The chosen action
        """
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.policy_network(state)
        action = torch.argmax(action_probs).item()
        return action

    def learn(self, state, action, reward, next_state):
        """
        Updates the agent's policy using the observed state, action, and reward.

        :param state: The current state
        :param action: The chosen action
        :param reward: The received reward
        :param next_state: The next state after the action
        """
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        # Compute loss and backpropagate
        action_probs = self.policy_network(state)
        predicted_value = action_probs[action]

        loss = -predicted_value * reward  # Using negative loss (policy gradient)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # Dynamically get noise_dim and output_dim
    noise_dim, output_dim = get_dynamic_dims()

    # Initialize GAN, RL agent, and attack environment with dynamically calculated dims
    gan_model = ThreatGenerator(noise_dim=noise_dim, output_dim=output_dim)  # Instantiate with dynamic args
    attack_env = ThreatMitigationEnv(generator=gan_model, noise_dim=noise_dim, output_dim=output_dim)  # Initialize environment
    rl_agent = RLAgent()  # Instantiate RL Agent

    # Set up and start continuous training pipeline
    pipeline = ContinuousTrainingPipeline(gan_model, rl_agent, attack_env, num_epochs=100)
    pipeline.continuous_training()
