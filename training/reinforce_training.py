import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Categorical
from stable_baselines3.common.monitor import Monitor

# Add parent directory to path to import our environment
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment.rendering import GarbageCollectionEnv

class PolicyNetwork(nn.Module):
    """A simple policy network for the REINFORCE agent."""
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        return self.network(obs)

class REINFORCEAgent:
    """A custom REINFORCE agent."""
    def __init__(self, env, lr=1e-4, gamma=0.99, device='cpu'):
        self.env = env
        self.gamma = gamma
        self.device = device
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        
        self.policy_net = PolicyNetwork(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.log_probs = []
        self.rewards = []

    def select_action(self, obs):
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        probs = self.policy_net(obs_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def update_policy(self):
        if not self.rewards:
            return 0.0

        returns = []
        g = 0
        for r in reversed(self.rewards):
            g = r + self.gamma * g
            returns.insert(0, g)

        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = [-log_prob * G for log_prob, G in zip(self.log_probs, returns)]
        
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        loss_val = policy_loss.item()
        self.log_probs = []
        self.rewards = []
        return loss_val

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))

def main():
    """Main function to train the REINFORCE model."""
    print("--- REINFORCE Training ---")
    
    device = 'cpu'
    print(f"Using device: {device}")
    
    log_dir = "logs/reinforce"
    model_dir = "models/reinforce"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    env = GarbageCollectionEnv(render_mode=None)
    env = Monitor(env, log_dir)
    
    agent = REINFORCEAgent(env, lr=1e-4, gamma=0.99, device=device)

    num_episodes = 5000
    max_steps_per_episode = 250
    print(f"Training for {num_episodes:,} episodes...")

    start_time = time.time()
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            
            for _ in range(max_steps_per_episode):
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                agent.rewards.append(reward)
                episode_reward += reward
                done = terminated or truncated
                if done:
                    break
            
            agent.update_policy()
            
            if (episode + 1) % 100 == 0:
                # This will read the monitor file for the average reward
                monitor_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("monitor.csv")]
                if monitor_files:
                    df = pd.read_csv(monitor_files[-1], skiprows=1)
                    if not df.empty:
                        avg_reward = df['r'].tail(100).mean()
                        print(f"Episode {episode+1}/{num_episodes} | Average Reward (last 100): {avg_reward:.2f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    
    training_time = time.time() - start_time
    print(f"\nTraining finished in {training_time/60:.1f} minutes.")
    
    model_path = os.path.join(model_dir, "reinforce_model.pth")
    agent.save(model_path)
    print(f"Final model saved to: {model_path}")
    
    env.close()
    
    generate_training_analysis(log_dir, model_dir)
    test_trained_model(model_path, device)
    print("\n--- REINFORCE Training Complete ---")

def generate_training_analysis(log_dir, model_dir):
    """Generates and saves a plot of the training progress."""
    monitor_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("monitor.csv")]
    if not monitor_files:
        monitor_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if "monitor" in f and f.endswith(".csv")]
        if not monitor_files:
            print("Could not find monitor file for analysis.")
            return

    df = pd.read_csv(monitor_files[-1], skiprows=1)
    if df.empty:
        print("Monitor file is empty, skipping analysis.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df['r'].rolling(window=50).mean())
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward (rolling window 50)")
    plt.title("REINFORCE Training Progress")
    plt.grid(True)
    
    save_path = os.path.join(model_dir, "reinforce_training_analysis.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Training analysis plot saved to: {save_path}")

def test_trained_model(model_path, device):
    """Tests the performance of the trained model."""
    print("\n--- Testing Trained Model ---")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    test_env = GarbageCollectionEnv(render_mode=None)
    agent = REINFORCEAgent(test_env, device=device)
    agent.load(model_path)
    agent.policy_net.eval() # Set to evaluation mode
    
    num_episodes = 10
    total_rewards = []
    
    for i in range(num_episodes):
        obs, _ = test_env.reset()
        done = False
        episode_reward = 0
        with torch.no_grad():
            while not done:
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, _ = test_env.step(action)
                episode_reward += reward
                done = terminated or truncated
        total_rewards.append(episode_reward)
        print(f"Test Episode {i+1}/{num_episodes} | Reward: {episode_reward:.2f}")
    
    print(f"\nAverage test reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
    test_env.close()

if __name__ == "__main__":
    main()
