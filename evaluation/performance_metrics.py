# performance_metrics.py
"""
Comprehensive performance evaluation for RL agents
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import torch
from stable_baselines3 import DQN, PPO, A2C

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment.rendering import GarbageCollectionEnv

class PerformanceEvaluator:
    def __init__(self, env_class=GarbageCollectionEnv):
        self.env_class = env_class
        self.results = {}
        
    def evaluate_agent(self, model_path, model_type, n_episodes=50, render=False):
        """
        Comprehensive evaluation of a trained agent
        
        Args:
            model_path: Path to trained model
            model_type: 'dqn', 'ppo', 'a2c', or 'reinforce'
            n_episodes: Number of evaluation episodes
            render: Whether to render episodes
        """
        print(f"\nEvaluating {model_type.upper()} agent...")
        
        # Load model
        if model_type == 'dqn':
            model = DQN.load(model_path)
        elif model_type == 'ppo':
            model = PPO.load(model_path)
        elif model_type == 'a2c':
            model = A2C.load(model_path)
        elif model_type == 'reinforce':
            # Custom REINFORCE loading would go here
            return self._evaluate_reinforce(model_path, n_episodes, render)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create environment
        env = self.env_class(render_mode="human" if render else None)
        
        # Evaluation metrics
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        garbage_collected_total = 0
        action_distribution = {i: 0 for i in range(6)}  # 6 actions
        exploration_ratios = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            actions_taken = []
            
            done = False
            while not done and episode_length < 200:  # Max episode length
                action, _ = model.predict(obs, deterministic=True)
                # Convert action to int if it's a numpy array
                if isinstance(action, np.ndarray):
                    action = int(action.item())
                else:
                    action = int(action)
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                actions_taken.append(action)
                action_distribution[action] += 1
                done = terminated or truncated
                
                if render:
                    env.render()
            
            # Calculate exploration ratio (unique actions / total actions)
            unique_actions = len(set(actions_taken))
            exploration_ratio = unique_actions / len(actions_taken) if actions_taken else 0
            exploration_ratios.append(exploration_ratio)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Check success (episode completed with positive reward)
            if info.get('success', False) or episode_reward > 3:
                success_count += 1
            
            # Track garbage collection if available
            if hasattr(env, 'garbage_collected'):
                garbage_collected_total += env.garbage_collected
        
        env.close()
        
        # Calculate comprehensive metrics
        metrics = {
            'model_type': model_type,
            'n_episodes': n_episodes,
            'average_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'average_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': (success_count / n_episodes) * 100,
            'average_exploration': np.mean(exploration_ratios),
            'action_distribution': action_distribution,
            'convergence_stability': np.std(episode_rewards[-10:]) if len(episode_rewards) >= 10 else float('inf'),
            'garbage_collected_avg': garbage_collected_total / n_episodes if garbage_collected_total > 0 else 0,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
        self.results[model_type] = metrics
        return metrics
    
    def _evaluate_reinforce(self, model_path, n_episodes, render):
        """Evaluate REINFORCE agent (placeholder for custom implementation)"""
        # This would load and evaluate your custom REINFORCE agent
        # For now, return dummy metrics
        return {
            'model_type': 'reinforce',
            'n_episodes': n_episodes,
            'average_reward': 0.0,
            'success_rate': 0.0,
            'note': 'REINFORCE evaluation not implemented'
        }
    
    def compare_agents(self, save_path="evaluation/comparison_results.json"):
        """Compare performance across all evaluated agents"""
        if not self.results:
            print("No agents evaluated yet!")
            return
        
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        
        # Create comparison table
        comparison_data = []
        for model_type, metrics in self.results.items():
            comparison_data.append({
                'Algorithm': model_type.upper(),
                'Avg Reward': f"{metrics['average_reward']:.2f} ± {metrics['std_reward']:.2f}",
                'Success Rate': f"{metrics['success_rate']:.1f}%",
                'Avg Length': f"{metrics['average_length']:.1f}",
                'Exploration': f"{metrics['average_exploration']:.3f}",
                'Stability': f"{metrics['convergence_stability']:.3f}"
            })
        
        # Print comparison table
        for data in comparison_data:
            print(f"{data['Algorithm']:12} | {data['Avg Reward']:15} | {data['Success Rate']:12} | {data['Avg Length']:10} | {data['Exploration']:11} | {data['Stability']:9}")
        
        # Save results
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate plots
        self._generate_comparison_plots()
        
        return comparison_data
    
    def _generate_comparison_plots(self):
        """Generate comparison visualization plots"""
        if len(self.results) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RL Algorithms Performance Comparison', fontsize=16)
        
        algorithms = list(self.results.keys())
        
        # Plot 1: Average Rewards
        avg_rewards = [self.results[alg]['average_reward'] for alg in algorithms]
        std_rewards = [self.results[alg]['std_reward'] for alg in algorithms]
        
        axes[0,0].bar(algorithms, avg_rewards, yerr=std_rewards, capsize=5)
        axes[0,0].set_title('Average Reward per Algorithm')
        axes[0,0].set_ylabel('Average Reward')
        
        # Plot 2: Success Rates
        success_rates = [self.results[alg]['success_rate'] for alg in algorithms]
        axes[0,1].bar(algorithms, success_rates)
        axes[0,1].set_title('Success Rate per Algorithm')
        axes[0,1].set_ylabel('Success Rate (%)')
        
        # Plot 3: Episode Length Distribution
        for alg in algorithms:
            if 'episode_lengths' in self.results[alg]:
                axes[1,0].hist(self.results[alg]['episode_lengths'], alpha=0.6, label=alg.upper(), bins=20)
        axes[1,0].set_title('Episode Length Distribution')
        axes[1,0].set_xlabel('Episode Length')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        
        # Plot 4: Reward Distribution
        for alg in algorithms:
            if 'episode_rewards' in self.results[alg]:
                axes[1,1].hist(self.results[alg]['episode_rewards'], alpha=0.6, label=alg.upper(), bins=20)
        axes[1,1].set_title('Reward Distribution')
        axes[1,1].set_xlabel('Episode Reward')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        # Save plots
        os.makedirs("evaluation/plots", exist_ok=True)
        plt.savefig("evaluation/plots/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Performance comparison plots saved to evaluation/plots/")

def main():
    """Main evaluation function"""
    evaluator = PerformanceEvaluator()
    
    # Define model paths
    models_to_evaluate = [
        ("training/models/dqn/dqn_model_final.zip", "dqn"),
        ("training/models/ppo/ppo_model_final.zip", "ppo"),
        ("training/models/a2c/a2c_model_final.zip", "a2c"),
    ]
    
    # Evaluate each model
    for model_path, model_type in models_to_evaluate:
        if os.path.exists(model_path):
            try:
                evaluator.evaluate_agent(model_path, model_type, n_episodes=30)
            except Exception as e:
                print(f"Error evaluating {model_type}: {e}")
        else:
            print(f"Model not found: {model_path}")
    
    # Generate comparison
    evaluator.compare_agents()

if __name__ == "__main__":
    main()
