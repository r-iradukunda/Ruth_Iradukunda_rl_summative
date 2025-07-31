# reinforce_training.py

import os
import sys

# Add the project root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from environment.rendering import GarbageCollectionEnv
from stable_baselines3.common.monitor import Monitor

class PolicyNetwork(nn.Module):
    def __init__(self, obs_space, act_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        
        # Process grid observations
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate conv output size
        with torch.no_grad():
            dummy_grid = torch.zeros(1, 1, obs_space['grid'].shape[0], obs_space['grid'].shape[1])
            conv_out_size = self.conv(dummy_grid).shape[1]
        
        # Combine features
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + 3, hidden_dim),  # +3 for agent_pos (2) and carrying (1)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        # Process grid through CNN
        grid = torch.FloatTensor(obs['grid']).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        grid_features = self.conv(grid)
        
        # Process other features
        agent_pos = torch.FloatTensor(obs['agent_pos'])
        carrying = torch.FloatTensor([float(obs['carrying'])])
        
        # Combine features
        combined = torch.cat([grid_features.squeeze(0), agent_pos, carrying])
        return self.fc(combined)

class REINFORCEAgent:
    def __init__(self, env, lr=3e-4, gamma=0.99, device='auto'):
        self.env = env
        self.gamma = gamma
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize network
        self.policy_net = PolicyNetwork(
            env.observation_space,
            env.action_space.n
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, obs):
        with torch.no_grad():
            probs = self.policy_net(obs)
        
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def update_policy(self):
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []
        
        return policy_loss.item()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")

def train_reinforce(num_episodes=1000, eval_freq=50):
    # Create directories
    model_dir = os.path.join(os.path.dirname(__file__), "models", "reinforce")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create environments
    env = GarbageCollectionEnv(render_mode=None)
    env = Monitor(env)
    
    eval_env = GarbageCollectionEnv(render_mode=None)
    eval_env = Monitor(eval_env)

    # Initialize agent
    agent = REINFORCEAgent(env)

    # Training loop
    best_eval_reward = float('-inf')
    reward_history = []
    
    print("Starting REINFORCE training...")
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.rewards.append(reward)
                episode_reward += reward
                obs = next_obs

            loss = agent.update_policy()
            reward_history.append(episode_reward)

            # Evaluation
            if (episode + 1) % eval_freq == 0:
                eval_rewards = []
                for _ in range(5):  # 5 evaluation episodes
                    eval_obs, _ = eval_env.reset()
                    eval_reward = 0
                    eval_done = False
                    
                    while not eval_done:
                        with torch.no_grad():
                            eval_action = agent.select_action(eval_obs)
                        eval_obs, r, terminated, truncated, _ = eval_env.step(eval_action)
                        eval_reward += r
                        eval_done = terminated or truncated
                    
                    eval_rewards.append(eval_reward)
                
                mean_eval_reward = np.mean(eval_rewards)
                print(f"Episode {episode + 1}: Training reward = {episode_reward:.2f}, "
                      f"Evaluation reward = {mean_eval_reward:.2f}, Loss = {loss:.2e}")

                # Save best model as reinforce_model
                if mean_eval_reward > best_eval_reward:
                    best_eval_reward = mean_eval_reward
                    agent.save(os.path.join(model_dir, "reinforce_model"))

            # Save checkpoint
            if (episode + 1) % 100 == 0:
                checkpoint_dir = os.path.join(model_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                agent.save(os.path.join(checkpoint_dir, f"reinforce_model_checkpoint_{episode+1}"))

        # Save final model
        final_model_path = os.path.join(model_dir, "reinforce_model")
        agent.save(final_model_path)
        print("Training complete!")
        print(f"Best evaluation reward: {best_eval_reward:.2f}")
        print(f"Final model saved to: {final_model_path}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise
    finally:
        env.close()
        eval_env.close()

    return reward_history

if __name__ == "__main__":
    train_reinforce()
