# actor_critic_training.py

import os
import sys
import time

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
    def _init():
        env = GarbageCollectionEnv(render_mode=render_mode)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_actor_critic():
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Create log/model directories
    os.makedirs("models/a2c", exist_ok=True)
    os.makedirs("logs/a2c", exist_ok=True)

    # Vectorized environments (reduced for CPU)
    n_envs = 2
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(n_envs, render_mode=None)])

    # Define model
    model = A2C(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=7e-4,
        n_steps=16,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        rms_prop_eps=1e-5,
        use_rms_prop=True,
        normalize_advantage=True,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128],
                vf=[128, 128]
            ),
            optimizer_class=torch.optim.RMSprop,
            optimizer_kwargs=dict(
                alpha=0.99,
                eps=1e-5
            )
        ),
        verbose=0,
        tensorboard_log="logs/a2c/",
        device=device
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/a2c/best_model",
        log_path="logs/a2c/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=1,
        verbose=0
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,  # Save once at end
        save_path="models/a2c/checkpoints/",
        name_prefix="a2c_model",
        save_vecnormalize=True
    )

    print("Starting A2C training...")

    try:
        total_timesteps = 5000  # Fast for debugging

        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
            log_interval=1000
        )

        model_path = "models/a2c/a2c_model"
        model.save(model_path)
        print(f"Model saved to {model_path}")

        # Final evaluation (1 episode only)
        print("\nEvaluating final model...")
        obs = eval_env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            episode_reward += rewards[0]
            done = dones[0]

        print(f"Final evaluation reward: {episode_reward:.2f}")

    except Exception as e:
        print(f"Training error: {e}")
        if device.type == "cuda":
            print(f"GPU Memory at error: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        raise
    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    train_actor_critic()
