import os
import sys
import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from environment.rendering import GarbageCollectionEnv
import torch

# Import REINFORCE model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.reinforce_training import ReinforceAgent  # Import the REINFORCE agent

def load_model(model_path):
    """
    Load a trained model based on the file path.
    Automatically determines the algorithm type from the path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Determine the algorithm type from the path
    if "dqn" in model_path.lower():
        model = DQN.load(model_path)
    elif "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path)
    elif "reinforce" in model_path.lower():
        # For REINFORCE, we need to create an environment instance to get the correct dimensions
        temp_env = GarbageCollectionEnv()
        model = ReinforceAgent(temp_env)
        model.load(model_path)
        temp_env.close()
    else:
        raise ValueError("Could not determine algorithm type from model path. Please use a path containing 'dqn', 'ppo', 'a2c', or 'reinforce'")
    
    return model

def run_model(model_path, num_episodes=5, render=True):
    """
    Run a trained model in the environment.
    
    Args:
        model_path (str): Path to the saved model file
        num_episodes (int): Number of episodes to run
        render (bool): Whether to render the environment
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load the model
        model = load_model(model_path)
        model.set_parameters(device=device)
        
        # Create and wrap the environment
        env = GarbageCollectionEnv(render_mode="human" if render else None)
        
        total_reward = 0
        
        # Run episodes
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            print(f"\nStarting Episode {episode + 1}")
            
            while not done:
                # Get action from model
                if isinstance(model, ReinforceAgent):
                    action = model.select_action(obs, deterministic=True)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                
                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step += 1
                
                if render:
                    env.render()
            
            print(f"Episode {episode + 1} finished after {step} steps with reward {episode_reward:.2f}")
            total_reward += episode_reward
        
        avg_reward = total_reward / num_episodes
        print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    # Example usage
    # You can replace these paths with any of your trained models
    model_paths = {
        "DQN": "training/models/dqn/dqn_model",
        "PPO": "training/models/ppo/ppo_model",
        "A2C": "training/models/a2c/a2c_model",
        "REINFORCE": "training/models/reinforce/reinforce_model",
    }
    
    # Ask user which model to run
    print("Available models:")
    for idx, (name, path) in enumerate(model_paths.items(), 1):
        print(f"{idx}. {name}")
    
    try:
        choice = int(input("\nSelect a model to run (enter number): "))
        if 1 <= choice <= len(model_paths):
            selected_model = list(model_paths.values())[choice - 1]
            num_episodes = int(input("Enter number of episodes to run: "))
            render = input("Enable rendering? (y/n): ").lower() == 'y'
            
            print(f"\nRunning model: {selected_model}")
            run_model(selected_model, num_episodes, render)
        else:
            print("Invalid selection!")
    except ValueError:
        print("Please enter a valid number!")
