import os
import sys
import time
import numpy as np
import argparse
from pathlib import Path
import pygame
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.rendering import GarbageCollectionEnv
from stable_baselines3 import DQN, PPO, A2C

# Import REINFORCE components
try:
    from training.reinforce_training import REINFORCEAgent
except ImportError:
    print("Warning: REINFORCE training module not found")
    REINFORCEAgent = None

class GamePlayer:
    """Main game player that loads and uses any trained RL model"""
    
    def __init__(self):
        self.env = None
        self.model = None
        self.model_path = None
        self.algorithm = None
        
    def detect_algorithm_from_path(self, model_path):
        """Detect algorithm type from model file path"""
        model_path_lower = model_path.lower()
        
        if 'dqn' in model_path_lower:
            return 'DQN'
        elif 'ppo' in model_path_lower:
            return 'PPO'
        elif 'a2c' in model_path_lower:
            return 'A2C'
        elif 'reinforce' in model_path_lower or model_path_lower.endswith('.pth'):
            return 'REINFORCE'
        elif model_path_lower.endswith('.zip'):
            # Default to DQN for .zip files if algorithm can't be determined
            print("Warning: Cannot detect algorithm from path, defaulting to DQN")
            return 'DQN'
        else:
            raise ValueError(f"Cannot detect algorithm from path: {model_path}")
    
    def load_model(self, model_path=None):
        """Load a trained model - automatically detects algorithm type"""
        
        # Use default DQN model path if none provided
        if not model_path:
            model_path = 'training/models/dqn/dqn_model.zip'
        
        self.model_path = model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Detect algorithm from path
        self.algorithm = self.detect_algorithm_from_path(model_path)
        print(f"Detected algorithm: {self.algorithm}")
        print(f"Loading {self.algorithm} model from: {model_path}")
        
        # Create environment for model loading
        temp_env = GarbageCollectionEnv(render_mode=None)
        
        try:
            if self.algorithm == 'DQN':
                self.model = DQN.load(model_path, env=temp_env)
            elif self.algorithm == 'PPO':
                self.model = PPO.load(model_path, env=temp_env)
            elif self.algorithm == 'A2C':
                self.model = A2C.load(model_path, env=temp_env)
            elif self.algorithm == 'REINFORCE':
                if REINFORCEAgent is None:
                    raise ImportError("REINFORCE training module not available")
                self.model = REINFORCEAgent(temp_env)
                self.model.load(model_path)
            
            temp_env.close()
            print(f"SUCCESS: {self.algorithm} model loaded successfully!")
            
        except Exception as e:
            temp_env.close()
            raise Exception(f"Failed to load {self.algorithm} model: {e}")
    
    def play_game(self, render_mode="human", max_steps=500, episodes=1, exploration_strategy="mixed"):
        """Play the game using the loaded model"""
        if self.model is None:
            raise ValueError("No model loaded! Please load a model first.")
        
        print(f"\nStarting {self.algorithm} agent ({exploration_strategy} exploration)")
        print(f"Episodes: {episodes} | Max steps: {max_steps}")
        print("="*50)
        
        total_rewards = []
        total_steps = []
        
        for episode in range(episodes):
            # Create environment for this episode
            self.env = GarbageCollectionEnv(render_mode=render_mode)
            
            print(f"\nEpisode {episode + 1}/{episodes}")
            print("-" * 30)
            
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            # Render initial state
            if render_mode == "human":
                self.env.render()
            
            print(f"Episode {episode + 1}/{episodes} - Environment ready")
            
            if render_mode == "human":
                time.sleep(1)  # Brief pause to see initial state
            
            # Tracking variables for debugging
            action_history = []
            position_history = []
            stuck_counter = 0
            last_positions = []
            
            try:
                for step in range(max_steps):
                    # Handle pygame events to keep window responsive
                    if render_mode == "human":
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                print("\nWindow closed by user")
                                return
                    
                    # Get action from model based on algorithm type and exploration strategy
                    if self.algorithm in ['DQN', 'PPO', 'A2C']:
                        # Apply exploration strategy
                        if exploration_strategy == 'deterministic':
                            action, _ = self.model.predict(obs, deterministic=True)
                        elif exploration_strategy == 'stochastic':
                            action, _ = self.model.predict(obs, deterministic=False)
                        else:  # mixed strategy
                            if step < 50:  # First 50 steps deterministic
                                action, _ = self.model.predict(obs, deterministic=True)
                            else:  # Then allow some exploration
                                action, _ = self.model.predict(obs, deterministic=False)
                        
                        if hasattr(action, 'item'):
                            action = action.item()
                    elif self.algorithm == 'REINFORCE':
                        # Custom REINFORCE model
                        obs_tensor = torch.FloatTensor(obs).to(self.model.device)
                        with torch.no_grad():
                            probs = self.model.policy_net(obs_tensor)
                            # Apply exploration strategy
                            if exploration_strategy == 'deterministic':
                                action = torch.argmax(probs).item()
                            elif exploration_strategy == 'stochastic':
                                action = torch.multinomial(probs, 1).item()
                            else:  # mixed strategy
                                if step < 50:
                                    action = torch.argmax(probs).item()
                                else:
                                    action = torch.multinomial(probs, 1).item()
                    else:
                        raise ValueError(f"Unknown algorithm: {self.algorithm}")
                    
                    # Track agent behavior for debugging
                    action_history.append(action)
                    current_pos = (self.env.agent_pos[0], self.env.agent_pos[1])
                    position_history.append(current_pos)
                    
                    # Check if agent is stuck (repeating same position)
                    last_positions.append(current_pos)
                    if len(last_positions) > 10:
                        last_positions.pop(0)
                        if len(set(last_positions)) <= 3:  # Only 3 or fewer unique positions in last 10 steps
                            stuck_counter += 1
                        else:
                            stuck_counter = 0
                    
                    # If stuck for too long, add some randomness
                    if stuck_counter > 20:
                        print(f"   Adding random action (agent stuck)")
                        action = np.random.randint(0, 6)  # Random action
                        stuck_counter = 0
                        last_positions.clear()
                    
                    # Take action
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    # Render the environment to update the GUI
                    if render_mode == "human":
                        self.env.render()
                    
                    episode_reward += reward
                    episode_steps += 1
                    obs = next_obs
                    
                    # Print action and reward info with more details
                    action_names = ['Up', 'Down', 'Left', 'Right', 'Pick up', 'Drop']
                    if step % 10 == 0 or reward != 0:  # Print every 10 steps or when reward changes
                        carrying = "+" if self.env.carrying else "-"
                        print(f"Step {step+1:3d}: {action_names[action]:8s} | {current_pos} | {carrying} | R:{reward:+5.1f} | Total:{episode_reward:+6.1f}")
                    
                    # Add small delay for human rendering
                    if render_mode == "human":
                        time.sleep(0.1)  # Faster rendering
                    
                    if terminated or truncated:
                        break
                
                # Episode summary with behavior analysis
                total_rewards.append(episode_reward)
                total_steps.append(episode_steps)
                
                print(f"\nEpisode {episode + 1}: Reward {episode_reward:+.1f} | Steps {episode_steps} | {'Completed' if terminated else 'Time limit'}")
                
                # Simplified behavior analysis
                if len(action_history) > 0:
                    action_names = ['Up', 'Down', 'Left', 'Right', 'Pick up', 'Drop']
                    movement_actions = sum(action_history[i] for i in range(len(action_history)) if action_history[i] < 4)
                    pickup_drop_actions = sum(1 for i in action_history if i >= 4)
                    
                    unique_positions = len(set(position_history))
                    exploration_ratio = unique_positions / len(position_history) if position_history else 0
                    
                    print(f"   Movement: {movement_actions} | Pickup/Drop: {pickup_drop_actions} | Exploration: {exploration_ratio:.0%}")
                    
                    if exploration_ratio < 0.3:
                        print(f"   Low exploration - try different algorithm or retrain")
                
                if render_mode == "human" and episodes > 1:
                    input("Press Enter for next episode...")
                
            except KeyboardInterrupt:
                print("\n\nGame interrupted by user")
                break
            finally:
                if self.env:
                    self.env.close()
        
        # Final statistics
        if total_rewards:
            print("\n" + "="*50)
            print("GAME STATISTICS")
            print("="*50)
            print(f"Algorithm: {self.algorithm}")
            print(f"Episodes: {len(total_rewards)} | Avg Reward: {np.mean(total_rewards):+.1f}")
            print(f"Best: {max(total_rewards):+.1f} | Total Steps: {sum(total_steps)}")
            
            # Performance evaluation
            avg_reward = np.mean(total_rewards)
            if avg_reward > 10:
                print("Performance: Excellent!")
            elif avg_reward > 0:
                print("Performance: Good!")
            elif avg_reward > -50:
                print("Performance: Needs improvement")
            else:
                print("Performance: Requires retraining")
                
            print("="*50)

def list_available_models():
    """List all available trained models"""
    print("\nAvailable Models:")
    print("-" * 30)
    
    model_info = [
        ('DQN', 'training/models/dqn/dqn_model.zip', 'Deep Q-Network'),
        ('PPO', 'training/models/ppo/ppo_model.zip', 'Proximal Policy Optimization'),
        ('A2C', 'training/models/a2c/a2c_model.zip', 'Advantage Actor-Critic'),
        ('REINFORCE', 'training/models/reinforce/reinforce_model.pth', 'Policy Gradient')
    ]
    
    available_models = []
    for algo, path, description in model_info:
        if os.path.exists(path):
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"[FOUND]    {algo:9s} - {description:25s} ({file_size:.1f} MB)")
            print(f"           Path: {path}")
            available_models.append(algo)
        else:
            print(f"[MISSING]  {algo:9s} - {description:25s} (Not trained)")
            print(f"           Expected: {path}")
        print()
    
    print("-" * 30)
    return available_models

def main():
    """Main function to run the game"""
    parser = argparse.ArgumentParser(
        description='Garbage Collection AI Demonstration - Watch trained RL agents in action!',
        epilog='Example: python main.py --exploration mixed --episodes 3'
    )
    parser.add_argument('--model', '-m', type=str, 
                        help='Path to model file (auto-detects and uses first available if not specified)')
    parser.add_argument('--episodes', '-e', type=int, default=1, 
                        help='Number of episodes to play (default: 1)')
    parser.add_argument('--steps', '-s', type=int, default=500, 
                        help='Max steps per episode (default: 500)')
    parser.add_argument('--render', '-r', type=str, choices=['human', 'rgb_array', 'none'], 
                        default='human', help='Render mode (default: human)')
    parser.add_argument('--list', '-l', action='store_true', 
                        help='List all available models and exit')
    parser.add_argument('--exploration', type=str, choices=['deterministic', 'stochastic', 'mixed'], 
                        default='mixed', 
                        help='Exploration strategy (default: mixed) - mixed = deterministic first 50 steps, then stochastic')
    
    args = parser.parse_args()
    
    print("GARBAGE COLLECTION AI")
    print("=" * 30)
    print("Trained RL agent demonstration")
    print("Supports: DQN, PPO, A2C, REINFORCE")
    print("=" * 30)
    print(f"Default exploration: {args.exploration} strategy")
    print()
    
    # List models if requested
    if args.list:
        available_models = list_available_models()
        print("\nQuick start examples:")
        print("  python main.py                    # Auto-select first available model")
        print("  python main.py --episodes 3       # Run 3 episodes")
        print("  python main.py --exploration stochastic  # More random behavior")
        return
    
    # Check if any model exists
    print("Scanning for trained models...")
    available_models = list_available_models()
    
    if not available_models and not args.model:
        print("\nERROR: No trained models found!")
        print("Please train a model first using the training scripts:")
        print("  - python training/dqn_training.py")
        print("  - python training/ppo_training.py") 
        print("  - python training/a2c_training.py")
        print("  - python training/reinforce_training.py")
        return
    
    # Show available models if no specific model provided
    if not args.model and available_models:
        # Use first available model as default
        default_model_paths = {
            'DQN': 'training/models/dqn/dqn_model.zip',
            'PPO': 'training/models/ppo/ppo_model.zip',
            'A2C': 'training/models/a2c/a2c_model.zip',
            'REINFORCE': 'training/models/reinforce/reinforce_model.pth'
        }
        
        for algo in ['DQN', 'PPO', 'A2C', 'REINFORCE']:
            if algo in available_models:
                args.model = default_model_paths[algo]
                print(f"\nAuto-selected model: {args.model}")
                print(f"Using exploration strategy: {args.exploration}")
                break
    
    # Initialize game player
    try:
        player = GamePlayer()
        player.load_model(args.model)
        
        # Play the game
        render_mode = None if args.render == 'none' else args.render
        player.play_game(
            render_mode=render_mode,
            max_steps=args.steps,
            episodes=args.episodes,
            exploration_strategy=args.exploration
        )
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure the model file exists or train the model first.")
        print("Use --list to see available models.")
    except Exception as e:
        print(f"ERROR: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()
