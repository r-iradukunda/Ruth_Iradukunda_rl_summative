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

# REINFORCE Training for Garbage Collection Environment
# Optimized for GPU training with comprehensive logging

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        
        # Simple MLP for flattened observations
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        # obs is already flattened (shape: 147)
        return self.fc(obs)

class REINFORCEAgent:
    def __init__(self, env, lr=3e-4, gamma=0.99):
        self.env = env
        self.gamma = gamma
        
        # GPU optimization for REINFORCE
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = torch.device('cpu')
            print("CUDA not available, using CPU")
        
        # Initialize network with flattened observation space
        obs_dim = env.observation_space.shape[0]  # Should be 147
        self.policy_net = PolicyNetwork(
            obs_dim,
            env.action_space.n
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, obs):
        # Convert observation to tensor and move to GPU
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        
        # Don't use no_grad here - we need gradients for policy updates
        probs = self.policy_net(obs_tensor)
        
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def update_policy(self):
        if not self.rewards or not self.log_probs:
            return 0.0
            
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        
        # Normalize returns only if we have more than one return
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        if not policy_loss:
            return 0.0
            
        policy_loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []
        
        return policy_loss.item()

    def save(self, path, verbose=True):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        if verbose:
            print(f"Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")

def main():
    print("REINFORCE TRAINING - Policy Gradient")
    print("=" * 50)
    
    # REINFORCE benefits from GPU acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create log directories
    log_dir = "training/logs/reinforce"
    model_dir = "models/reinforce"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Directories created:")
    print(f"   Logs: {log_dir}")
    print(f"   Models: {model_dir}")

    # Create environment
    env = GarbageCollectionEnv(render_mode=None)
    env = Monitor(env, log_dir)
    
    print(f"Environment Details:")
    print(f"   Grid Size: 12x12")
    print(f"   Action Space: {env.action_space}")
    print(f"   Observation Space: {env.observation_space}")
    
    # REINFORCE Configuration
    print(f"\nREINFORCE Configuration:")
    config = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'hidden_dim': 256,
        'max_episodes': 2000,
        'max_steps_per_episode': 200,
        'device': device
    }
    
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Create REINFORCE agent
    print(f"\nCreating REINFORCE agent...")
    agent = REINFORCEAgent(env, lr=config['learning_rate'], gamma=config['gamma'])
    
    # Training configuration
    max_episodes = config['max_episodes']
    print(f"\nTraining Configuration:")
    print(f"   Max episodes: {max_episodes:,}")
    print(f"   Max steps per episode: {config['max_steps_per_episode']}")
    print(f"   Estimated time: ~{max_episodes // 10} minutes on GPU")
    
    # Start training
    print(f"\nStarting REINFORCE training...")
    start_time = time.time()
    
    episode_rewards = []
    episode_losses = []
    best_reward = float('-inf')
    
    try:
        for episode in range(max_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            
            # Collect episode trajectory
            for step in range(config['max_steps_per_episode']):
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                
                agent.rewards.append(reward)
                episode_reward += reward
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            # Update policy
            loss = agent.update_policy()
            episode_rewards.append(episode_reward)
            episode_losses.append(loss)
            
            # Track best performance
            if episode_reward > best_reward:
                best_reward = episode_reward
                # Save best model silently
                model_path = os.path.join(model_dir, "reinforce_model.pth")
                agent.save(model_path, verbose=False)
            
            # Progress reporting
            if (episode + 1) % 100 == 0:
                recent_rewards = episode_rewards[-100:]
                avg_reward = np.mean(recent_rewards)
                avg_loss = np.mean(episode_losses[-100:]) if episode_losses[-100:] else 0
                print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, "
                      f"Best Reward = {best_reward:.2f}, Avg Loss = {avg_loss:.4f}")
            
            # Early stopping if performance is good
            if len(episode_rewards) >= 100:
                recent_avg = np.mean(episode_rewards[-100:])
                if recent_avg > 10:  # Adjust threshold based on your reward scale
                    print(f"Early stopping at episode {episode + 1} - good performance achieved!")
                    break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"   Episodes completed: {len(episode_rewards)}")
        print(f"   Best reward achieved: {best_reward:.2f}")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        training_time = time.time() - start_time
        print(f"   Partial training time: {training_time:.2f} seconds")
    
    # Save final model
    final_model_path = os.path.join(model_dir, "reinforce_model.pth")
    agent.save(final_model_path, verbose=True)
    print(f"Final model saved to: {final_model_path}")
    
    # Close environment
    env.close()
    
    # Generate training analysis
    generate_training_analysis(log_dir, model_dir, episode_rewards, episode_losses)
    
    # Test the trained model
    test_trained_model(final_model_path, env)
    
    print(f"\nREINFORCE training complete!")
    print(f"   Model: {final_model_path}")
    print(f"   Logs: {log_dir}")

def generate_training_analysis(log_dir, model_dir, episode_rewards, episode_losses):
    """Generate comprehensive training analysis plots"""
    print(f"\nGenerating training analysis...")
    
    try:
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('REINFORCE Training Analysis', fontsize=16, fontweight='bold')
        
        # 1. Episode rewards
        episodes = range(len(episode_rewards))
        window_size = min(50, len(episode_rewards))
        if len(episode_rewards) >= window_size:
            moving_avg = pd.Series(episode_rewards).rolling(window=window_size, min_periods=1).mean()
        else:
            moving_avg = episode_rewards
            
        axes[0, 0].plot(episodes, episode_rewards, alpha=0.3, color='lightblue', label='Episode Reward')
        axes[0, 0].plot(episodes, moving_avg, color='darkblue', linewidth=2, label=f'Moving Average ({window_size})')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Cumulative reward
        cumulative_rewards = np.cumsum(episode_rewards)
        axes[0, 1].plot(episodes, cumulative_rewards, color='green', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Cumulative Reward')
        axes[0, 1].set_title('Learning Accumulation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Policy loss
        if episode_losses and any(loss is not None for loss in episode_losses):
            valid_losses = [loss for loss in episode_losses if loss is not None]
            loss_episodes = range(len(valid_losses))
            axes[1, 0].plot(loss_episodes, valid_losses, color='red', alpha=0.7, linewidth=1)
            if len(valid_losses) >= window_size:
                loss_moving_avg = pd.Series(valid_losses).rolling(window=window_size, min_periods=1).mean()
                axes[1, 0].plot(loss_episodes, loss_moving_avg, color='darkred', linewidth=2, label='Moving Average')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Policy Loss')
            axes[1, 0].set_title('Policy Loss Over Time')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No loss data available', 
                           transform=axes[1, 0].transAxes, ha='center', va='center')
            axes[1, 0].set_title('Policy Loss (No Data)')
        
        # 4. Performance statistics
        axes[1, 1].axis('off')
        
        # Calculate improvement
        if len(episode_rewards) >= 4:
            first_quarter = np.mean(episode_rewards[:len(episode_rewards)//4])
            last_quarter = np.mean(episode_rewards[3*len(episode_rewards)//4:])
            improvement = last_quarter - first_quarter
        else:
            first_quarter = episode_rewards[0] if episode_rewards else 0
            last_quarter = episode_rewards[-1] if episode_rewards else 0
            improvement = last_quarter - first_quarter
        
        stats_text = f"""
REINFORCE Training Statistics:

Episodes: {len(episode_rewards)}
Mean Reward: {np.mean(episode_rewards):.2f}
Best Reward: {max(episode_rewards):.2f}
Final Reward: {episode_rewards[-1]:.2f}

Standard Deviation: {np.std(episode_rewards):.2f}
Success Rate: {(np.array(episode_rewards) > 0).mean()*100:.1f}%

Learning Trend:
First 25%: {first_quarter:.2f}
Last 25%: {last_quarter:.2f}
Improvement: {improvement:.2f}
        """
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        # Save the analysis
        analysis_path = os.path.join(model_dir, "reinforce_training_analysis.png")
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training analysis saved to: {analysis_path}")
        
    except Exception as e:
        print(f"Error generating analysis: {e}")

def test_trained_model(model_path, env_template):
    """Test the trained model"""
    print(f"\nTesting trained model...")
    
    try:
        # Create test environment
        test_env = GarbageCollectionEnv(render_mode=None)
        
        # Create and load agent
        agent = REINFORCEAgent(test_env)
        agent.load(model_path)
        
        # Run demonstration episodes
        num_test_episodes = 5
        total_rewards = []
        
        print(f"Running {num_test_episodes} test episodes...")
        
        for episode in range(num_test_episodes):
            obs, _ = test_env.reset()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(200):  # Max steps per episode
                # Deterministic action selection for testing
                obs_tensor = torch.FloatTensor(obs).to(agent.device)
                with torch.no_grad():
                    probs = agent.policy_net(obs_tensor)
                    action = torch.argmax(probs).item()
                
                obs, reward, terminated, truncated, _ = test_env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
            print(f"   Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")
        
        test_env.close()
        
        # Print test results
        avg_reward = np.mean(total_rewards)
        print(f"\nTest Results:")
        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Best Episode: {max(total_rewards):.2f}")
        print(f"   Worst Episode: {min(total_rewards):.2f}")
        print(f"   Standard Deviation: {np.std(total_rewards):.2f}")
        
    except Exception as e:
        print(f"Error testing model: {e}")
        print("Model testing failed, but training was completed successfully.")

if __name__ == "__main__":
    main()
