# pg_training.py

import os
import sys

# Add the project root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from environment.rendering import GarbageCollectionEnv
import numpy as np
import torch

def make_env(rank, seed=0, render_mode=None):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = GarbageCollectionEnv(render_mode=render_mode)
        env = Monitor(env)  # Wrap with Monitor
        env.reset(seed=seed + rank)  # Seed each env differently
        return env
    return _init

def train_ppo():
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Create directories for logs and models
    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("../logs/ppo", exist_ok=True)

    # Set number of environments
    n_envs = 4  # Number of parallel environments
    
    # Create vectorized environments for training using the same wrapper type
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    
    # Create evaluation environment with the same wrapper type
    eval_env = DummyVecEnv([make_env(n_envs, render_mode=None)])  # Different seed for eval

    # Define PPO model with optimized hyperparameters
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256]
            ),
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=dict(eps=1e-5)
        ),
        verbose=1,
        tensorboard_log="../logs/ppo/",
        device=device
    )

    # Callbacks for evaluation and checkpoints
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/ppo/best_model",
        log_path="../logs/ppo/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/ppo/checkpoints/",
        name_prefix="ppo_model",
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    print("Starting PPO training...")
    try:
        # Train the model
        total_timesteps = 50000
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
            log_interval=100
        )

        # Save the final model
        final_model_path = "models/ppo/ppo_model"
        model.save(final_model_path)
        print(f"Model saved successfully to {final_model_path}")

        # Evaluate final model
        mean_reward = 0
        n_eval_episodes = 20
        
        print("\nEvaluating final model...")
        for episode in range(n_eval_episodes):
            obs = eval_env.reset()[0]
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
            mean_reward += episode_reward
            print(f"Episode {episode + 1}/{n_eval_episodes}: Reward = {episode_reward:.2f}")
        
        mean_reward /= n_eval_episodes
        print(f"\nFinal evaluation: mean reward = {mean_reward:.2f}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        if device.type == "cuda":
            print(f"GPU Memory at error: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        raise
    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    train_ppo()
