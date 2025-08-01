import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Add parent directory to path to import our environment
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment.rendering import GarbageCollectionEnv

# DQN Training for Garbage Collection Environment
# Optimized for GPU training with comprehensive logging

def main():
    print("DQN TRAINING - Deep Q-Network")
    print("=" * 50)
    
    # GPU optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create log directories
    log_dir = "logs/dqn"
    model_dir = "models/dqn"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Directories created:")
    print(f"   Logs: {log_dir}")
    print(f"   Models: {model_dir}")

    # Create environment
    def make_env():
        env = GarbageCollectionEnv(render_mode=None)  # No rendering during training
        env = Monitor(env, log_dir)  # Monitor for logging
        return env
    
    env = DummyVecEnv([make_env])  # Vectorize for SB3

    
    print(f"Environment Details:")
    print(f"   Grid Size: 12x12")
    print(f"   Action Space: {env.action_space}")
    print(f"   Observation Space: {env.observation_space}")
    
    # DQN Hyperparameters - optimized for our garbage collection task
    print(f"\nDQN Configuration:")
    hyperparams = {
        'policy': "MlpPolicy",
        'env': env,
        'learning_rate': 2e-4,  # Conservative learning rate
        'buffer_size': 50000,  # Large experience replay buffer
        'learning_starts': 1000,  # Start learning after some exploration
        'batch_size': 64,  # Stable gradient estimates
        'tau': 1.0,  # Hard target network updates
        'gamma': 0.995,  # Long-term reward consideration
        'train_freq': 4,  # Train every 4 steps
        'gradient_steps': 1,  # One gradient step per train
        'target_update_interval': 1000,  # Target network update frequency
        'exploration_fraction': 0.2,  # 20% of training for exploration
        'exploration_initial_eps': 1.0,  # Start with full exploration
        'exploration_final_eps': 0.05,  # End with minimal exploration
        'verbose': 0,  # Reduce verbose output
        'tensorboard_log': log_dir,
        'device': device,  # GPU/CPU selection
        'policy_kwargs': {
            'net_arch': [256, 256, 128],  # Deep network for complex decisions
            'activation_fn': torch.nn.ReLU
        }
    }
    
    for key, value in hyperparams.items():
        if key not in ['env', 'policy_kwargs']:
            print(f"   {key}: {value}")
    print(f"   network_architecture: {hyperparams['policy_kwargs']['net_arch']}")
    
    # Create DQN model
    print(f"\nCreating DQN model...")
    dqn_model = DQN(**hyperparams)

    
    # Training configuration
    total_timesteps = 30000  # Balanced training time
    print(f"\nTraining Configuration:")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Expected episodes: ~{total_timesteps // 200}")  # Assuming ~200 steps per episode
    print(f"   Estimated time: ~{total_timesteps // 1000} minutes on GPU")
    
    # Create callbacks for better training monitoring
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,  # Save every 5000 steps
        save_path=model_dir,
        name_prefix="dqn_checkpoint"
    )
    
    # Start training
    print(f"\nStarting DQN training...")
    start_time = time.time()
    
    try:
        dqn_model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"   Average: {training_time/total_timesteps*1000:.2f} ms per step")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        training_time = time.time() - start_time
        print(f"   Partial training time: {training_time:.2f} seconds")
    
    # Save the final model
    model_path = os.path.join(model_dir, "dqn_model.zip")
    dqn_model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Close training environment
    env.close()
    
    # Generate training analysis
    generate_training_analysis(log_dir, model_dir)
    
    # Test the trained model
    test_trained_model(model_path)
    
    print(f"\nDQN training complete!")
    print(f"   Model: {model_path}")
    print(f"   Logs: {log_dir}")

def generate_training_analysis(log_dir, model_dir):
    """Generate comprehensive training analysis plots"""
    print(f"\nGenerating training analysis...")
    
    # Look for monitor files
    monitor_files = []
    for file in os.listdir(log_dir):
        if file.startswith("monitor") and file.endswith(".csv"):
            monitor_files.append(os.path.join(log_dir, file))
    
    if not monitor_files:
        print(f"No monitor files found in {log_dir}")
        return
    
    # Use the first monitor file found
    monitor_file = monitor_files[0]
    print(f"Using monitor file: {monitor_file}")
    
    try:
        # Load training data - monitor files have a header comment line
        with open(monitor_file, 'r') as f:
            first_line = f.readline()
            
        # Skip header if it starts with #
        skiprows = 1 if first_line.startswith('#') else 0
        df = pd.read_csv(monitor_file, skiprows=skiprows)
        
        if df.empty:
            print(f"Monitor file is empty")
            return
            
        # Check for reward column (could be 'r' or 'reward')
        reward_col = None
        if 'r' in df.columns:
            reward_col = 'r'
        elif 'reward' in df.columns:
            reward_col = 'reward'
        else:
            print(f"No reward data found. Available columns: {list(df.columns)}")
            return
        
        print(f"Found {len(df)} episodes in monitor data")
        
        # Calculate metrics
        df['episode'] = range(len(df))
        df['avg_reward'] = df[reward_col].rolling(window=min(50, len(df)), min_periods=1).mean()
        df['cumulative_reward'] = df[reward_col].cumsum()
        
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Training Analysis', fontsize=16, fontweight='bold')
        
        # 1. Episode rewards
        axes[0, 0].plot(df['episode'], df[reward_col], alpha=0.3, color='lightblue', label='Episode Reward')
        axes[0, 0].plot(df['episode'], df['avg_reward'], color='darkblue', linewidth=2, label=f'Moving Average ({min(50, len(df))})')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Cumulative reward
        axes[0, 1].plot(df['episode'], df['cumulative_reward'], color='green', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Cumulative Reward')
        axes[0, 1].set_title('Learning Accumulation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Episode length
        length_col = None
        if 'l' in df.columns:
            length_col = 'l'
        elif 'length' in df.columns:
            length_col = 'length'
            
        if length_col:
            df['avg_length'] = df[length_col].rolling(window=min(50, len(df)), min_periods=1).mean()
            axes[1, 0].plot(df['episode'], df[length_col], alpha=0.3, color='orange', label='Episode Length')
            axes[1, 0].plot(df['episode'], df['avg_length'], color='red', linewidth=2, label='Moving Average')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Steps')
            axes[1, 0].set_title('Episode Length')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No episode length data available', 
                           transform=axes[1, 0].transAxes, ha='center', va='center')
            axes[1, 0].set_title('Episode Length (No Data)')
        
        # 4. Performance statistics
        axes[1, 1].axis('off')
        
        # Calculate improvement
        if len(df) >= 4:
            first_quarter = df[reward_col][:len(df)//4].mean()
            last_quarter = df[reward_col][3*len(df)//4:].mean()
            improvement = last_quarter - first_quarter
        else:
            first_quarter = df[reward_col].iloc[0] if len(df) > 0 else 0
            last_quarter = df[reward_col].iloc[-1] if len(df) > 0 else 0
            improvement = last_quarter - first_quarter
        
        stats_text = f"""
DQN Training Statistics:

Episodes: {len(df)}
Mean Reward: {df[reward_col].mean():.2f}
Best Reward: {df[reward_col].max():.2f}
Final Avg Reward: {df['avg_reward'].iloc[-1]:.2f}

Standard Deviation: {df[reward_col].std():.2f}
Success Rate: {(df[reward_col] > 0).mean()*100:.1f}%

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
        analysis_path = os.path.join(model_dir, "dqn_training_analysis.png")
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training analysis saved to: {analysis_path}")
        
    except Exception as e:
        print(f"Error generating analysis: {e}")
        print(f"Monitor file exists but couldn't be processed: {monitor_file}")

def test_trained_model(model_path):
    """Test the trained model"""
    print(f"\nTesting trained model...")
    
    try:
        # Create test environment
        test_env = GarbageCollectionEnv(render_mode=None)
        
        # Load trained model
        dqn_model = DQN.load(model_path)
        
        # Run demonstration episodes
        num_test_episodes = 5
        total_rewards = []
        
        print(f"Running {num_test_episodes} test episodes...")
        
        for episode in range(num_test_episodes):
            obs, _ = test_env.reset()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(200):  # Max steps per episode
                action, _ = dqn_model.predict(obs, deterministic=True)
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