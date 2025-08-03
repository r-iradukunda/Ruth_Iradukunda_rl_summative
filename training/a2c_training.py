import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Add parent directory to path to import our environment
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment.rendering import GarbageCollectionEnv

def main():
    """Main function to train the A2C model."""
    print("--- A2C Training ---")
    
    device = 'cpu'
    print(f"Using device: {device}")
    
    log_dir = "logs/a2c"
    model_dir = "models/a2c"
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
        'learning_rate': 1e-3,
        'n_steps': 16,
        'gamma': 0.99,
        'gae_lambda': 1.0,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_rms_prop': True,
        'verbose': 1,
        'tensorboard_log': log_dir,
        'device': device,
        'policy_kwargs': {
            'net_arch': [dict(pi=[128, 128], vf=[128, 128])],
            'activation_fn': torch.nn.ReLU
        }
    }
    
    a2c_model = A2C(**hyperparams)

    total_timesteps = 100_000
    print(f"Training for {total_timesteps:,} timesteps...")

    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path=model_dir,
        name_prefix="a2c_model"
    )
    
    start_time = time.time()
    try:
        a2c_model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    
    training_time = time.time() - start_time
    print(f"\nTraining finished in {training_time/60:.1f} minutes.")
    
    model_path = os.path.join(model_dir, "a2c_model_final.zip")
    a2c_model.save(model_path)
    print(f"Final model saved to: {model_path}")
    
    env.close()
    
    generate_training_analysis(log_dir, model_dir)
    test_trained_model(model_path)
    print("\n--- A2C Training Complete ---")

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
    plt.title("A2C Training Progress")
    plt.grid(True)
    
    save_path = os.path.join(model_dir, "a2c_training_analysis.png")
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
    model = A2C.load(model_path, env=test_env)
    
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
