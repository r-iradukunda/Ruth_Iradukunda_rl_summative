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
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# Add parent directory to path to import our environment
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment.rendering import GarbageCollectionEnv

class ProgressCallback(BaseCallback):
    """A custom callback to print training progress."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        self.current_episode_length += 1
        self.current_episode_reward += self.locals['rewards'][0]
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            if len(self.episode_rewards) % 20 == 0:
                avg_reward = np.mean(self.episode_rewards[-20:])
                print(f"Episodes: {len(self.episode_rewards):<5} | "
                      f"Avg Reward (last 20): {avg_reward:<8.2f} | "
                      f"Epsilon: {self.model.exploration_rate:.2f}")
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
        return True

def main():
    """Main function to train the DQN model."""
    print("--- DQN Training ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    log_dir = "logs/dqn"
    model_dir = "models/dqn"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    def make_env():
        env = GarbageCollectionEnv(render_mode=None)
        env = Monitor(env, log_dir)
        return env
    
    env = DummyVecEnv([make_env])

    # Hyperparameters optimized for better performance
    hyperparams = {
        'policy': "MlpPolicy",
        'env': env,
        'learning_rate': 5e-4,
        'buffer_size': 100_000,
        'learning_starts': 5000,
        'batch_size': 128,
        'gamma': 0.99,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 1000,
        'exploration_fraction': 0.4,
        'exploration_final_eps': 0.05,
        'verbose': 0, # Use custom callback for cleaner output
        'tensorboard_log': log_dir,
        'device': device,
        'policy_kwargs': {
            'net_arch': [256, 256],
            'activation_fn': torch.nn.ReLU
        }
    }
    
    dqn_model = DQN(**hyperparams)

    total_timesteps = 150_000
    print(f"Training for {total_timesteps:,} timesteps...")

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=model_dir,
        name_prefix="dqn_model"
    )
    progress_callback = ProgressCallback()
    
    start_time = time.time()
    try:
        dqn_model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, progress_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    
    training_time = time.time() - start_time
    print(f"\nTraining finished in {training_time/60:.1f} minutes.")
    
    model_path = os.path.join(model_dir, "dqn_model_final.zip")
    dqn_model.save(model_path)
    print(f"Final model saved to: {model_path}")
    
    env.close()
    
    generate_training_analysis(log_dir, model_dir)
    test_trained_model(model_path)
    print("\n--- DQN Training Complete ---")

def generate_training_analysis(log_dir, model_dir):
    """Generates and saves a plot of the training progress."""
    monitor_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("monitor.csv")]
    if not monitor_files:
        # Fallback for cases where monitor file has a timestamp
        monitor_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if "monitor" in f and f.endswith(".csv")]
        if not monitor_files:
            print("Could not find monitor file for analysis.")
            return

    # Read the latest monitor file
    df = pd.read_csv(monitor_files[-1], skiprows=1)
    if df.empty:
        print("Monitor file is empty, skipping analysis.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df['r'].rolling(window=50).mean())
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward (rolling window 50)")
    plt.title("DQN Training Progress")
    plt.grid(True)
    
    save_path = os.path.join(model_dir, "dqn_training_analysis.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Training analysis plot saved to: {save_path}")

def test_trained_model(model_path):
    """Tests the performance of the trained model."""
    print("\n--- Testing Trained Model ---")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    test_env = GarbageCollectionEnv(render_mode=None)
    model = DQN.load(model_path, env=test_env)
    
    num_episodes = 10
    total_rewards = []
    
    for i in range(num_episodes):
        obs, _ = test_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_rewards.append(episode_reward)
        print(f"Test Episode {i+1}/{num_episodes} | Reward: {episode_reward:.2f}")
    
    print(f"\nAverage test reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
    test_env.close()

if __name__ == "__main__":
    main()
