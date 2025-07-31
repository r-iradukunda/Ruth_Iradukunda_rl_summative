import os
import sys

# Add the project root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from environment.rendering import GarbageCollectionEnv
import numpy as np
import torch

def train_dqn():
    # Check for GPU availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    # Create and wrap the environment
    env = GarbageCollectionEnv(render_mode=None)
    env = Monitor(env)
    
    # Create eval environment
    eval_env = GarbageCollectionEnv(render_mode=None)
    eval_env = Monitor(eval_env)

    # Create directories for logs and models
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("../logs/dqn", exist_ok=True)

    # Define optimized hyperparameters for DQN
    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(
            net_arch=[128, 128, 64],  # Simplified architecture
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=dict(eps=1e-5)
        ),
        verbose=1,
        tensorboard_log="../logs/dqn/",
        device=device  # Explicitly set the device
    )

    # Callbacks for evaluation and checkpoints
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/dqn/best_model",
        log_path="../logs/dqn/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/dqn/checkpoints/",
        name_prefix="dqn_model",
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    print("Starting training...")
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
        model.save("models/dqn/dqn_model")
        print("Model saved successfully!")

        # Print GPU memory usage after training if using CUDA
        if device.type == "cuda":
            print(f"Final GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"Max GPU Memory Allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")

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

if __name__ == "__main__":
    train_dqn()
