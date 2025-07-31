# actor_critic_training.py

import os
import sys

# Add the project root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from environment.rendering import GarbageCollectionEnv
import numpy as np
import torch

def make_env(rank, seed=0, render_mode=None):
    """
    Utility function for environment creation
    """
    def _init():
        env = GarbageCollectionEnv(render_mode=render_mode)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_actor_critic():
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Create directories for logs and models
    os.makedirs("models/a2c", exist_ok=True)
    os.makedirs("../logs/a2c", exist_ok=True)

    # Create vectorized environments
    n_envs = 8  # A2C benefits from more parallel environments
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(n_envs, render_mode=None)])  # Separate env for evaluation

    # Define the A2C model with optimized hyperparameters
    model = A2C(
        policy="MultiInputPolicy",  # For dictionary observation space
        env=env,
        learning_rate=7e-4,
        n_steps=8,            # Number of steps before updating
        gamma=0.99,          # Discount factor
        gae_lambda=0.95,     # GAE parameter
        ent_coef=0.01,       # Entropy coefficient for exploration
        vf_coef=0.5,         # Value function coefficient
        max_grad_norm=0.5,   # Gradient clipping
        rms_prop_eps=1e-5,   # RMSprop epsilon
        use_rms_prop=True,   # Use RMSprop optimizer
        normalize_advantage=True,  # Normalize advantages
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128],    # Actor network
                vf=[128, 128]     # Critic network
            ),
            optimizer_class=torch.optim.RMSprop,
            optimizer_kwargs=dict(
                alpha=0.99,
                eps=1e-5
            )
        ),
        verbose=1,
        tensorboard_log="../logs/a2c/",
        device=device
    )

    # Callbacks for evaluation and checkpoints
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/a2c/best_model",
        log_path="../logs/a2c/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/a2c/checkpoints/",
        name_prefix="a2c_model",
        save_vecnormalize=True
    )

    print("Starting A2C training...")
    try:
        # Train the model
        total_timesteps = 50000  # A2C often needs more steps
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
            log_interval=100
        )

        # Save the final model
        final_model_path = "models/a2c/a2c_model"
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
    train_actor_critic()