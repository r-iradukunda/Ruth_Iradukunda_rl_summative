# visualization_system.py
"""
Comprehensive Visualization System for RL Algorithm Comparison
Creates plots, charts, and visual analysis for assignment requirements
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class RLVisualizationSystem:
    def __init__(self):
        self.results_data = {}
        self.plots_dir = "evaluation/plots"
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def load_results(self, results_file=None):
        """Load evaluation results from JSON file"""
        if results_file is None:
            # Find the most recent results file
            results_dir = "evaluation/results"
            if os.path.exists(results_dir):
                files = [f for f in os.listdir(results_dir) if f.startswith("rl_comparison_")]
                if files:
                    results_file = os.path.join(results_dir, sorted(files)[-1])
        
        if results_file and os.path.exists(results_file):
            with open(results_file, 'r') as f:
                self.results_data = json.load(f)
            print(f"Loaded results from: {results_file}")
        else:
            print("No results file found. Generating with sample data.")
            self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate sample data for demonstration purposes"""
        self.results_data = {
            'dqn': {
                'mean_reward': 4.2,
                'std_reward': 1.8,
                'success_rate': 75.0,
                'mean_exploration': 0.65,
                'performance_stability': 0.85,
                'episode_rewards': list(np.random.normal(4.2, 1.8, 30)),
                'action_distribution': {0: 45, 1: 38, 2: 42, 3: 40, 4: 25, 5: 30},
                'hyperparameters': {
                    'learning_rate': 3e-4,
                    'batch_size': 64,
                    'gamma': 0.99
                }
            },
            'ppo': {
                'mean_reward': 3.8,
                'std_reward': 1.5,
                'success_rate': 68.0,
                'mean_exploration': 0.72,
                'performance_stability': 0.62,
                'episode_rewards': list(np.random.normal(3.8, 1.5, 30)),
                'action_distribution': {0: 40, 1: 42, 2: 38, 3: 45, 4: 28, 5: 27},
                'hyperparameters': {
                    'learning_rate': 3e-4,
                    'n_steps': 1024,
                    'gamma': 0.99
                }
            },
            'a2c': {
                'mean_reward': 3.5,
                'std_reward': 2.1,
                'success_rate': 62.0,
                'mean_exploration': 0.58,
                'performance_stability': 1.2,
                'episode_rewards': list(np.random.normal(3.5, 2.1, 30)),
                'action_distribution': {0: 35, 1: 41, 2: 44, 3: 38, 4: 22, 5: 20},
                'hyperparameters': {
                    'learning_rate': 7e-4,
                    'n_steps': 16,
                    'gamma': 0.99
                }
            }
        }
    
    def create_performance_comparison(self):
        """Create comprehensive performance comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RL Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        
        algorithms = list(self.results_data.keys())
        
        # 1. Mean Reward Comparison
        ax1 = axes[0, 0]
        rewards = [self.results_data[alg]['mean_reward'] for alg in algorithms]
        errors = [self.results_data[alg]['std_reward'] for alg in algorithms]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars = ax1.bar(algorithms, rewards, yerr=errors, capsize=5, 
                      color=colors[:len(algorithms)], alpha=0.8)
        ax1.set_title('Average Reward per Episode')
        ax1.set_ylabel('Mean Reward')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, reward in zip(bars, rewards):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{reward:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Success Rate Comparison
        ax2 = axes[0, 1]
        success_rates = [self.results_data[alg]['success_rate'] for alg in algorithms]
        
        bars = ax2.bar(algorithms, success_rates, color=colors[:len(algorithms)], alpha=0.8)
        ax2.set_title('Success Rate')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars, success_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Exploration vs Stability
        ax3 = axes[1, 0]
        exploration = [self.results_data[alg]['mean_exploration'] for alg in algorithms]
        stability = [1/self.results_data[alg]['performance_stability'] for alg in algorithms]  # Inverse for better visualization
        
        scatter = ax3.scatter(exploration, stability, s=200, 
                            c=colors[:len(algorithms)], alpha=0.7)
        
        for i, alg in enumerate(algorithms):
            ax3.annotate(alg.upper(), (exploration[i], stability[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax3.set_xlabel('Exploration Score')
        ax3.set_ylabel('Stability Score (1/variance)')
        ax3.set_title('Exploration vs Stability Trade-off')
        ax3.grid(True, alpha=0.3)
        
        # 4. Learning Curves
        ax4 = axes[1, 1]
        for i, alg in enumerate(algorithms):
            rewards_history = self.results_data[alg]['episode_rewards']
            episodes = range(1, len(rewards_history) + 1)
            
            # Calculate moving average
            window_size = min(5, len(rewards_history)//3)
            if window_size > 1:
                moving_avg = pd.Series(rewards_history).rolling(window=window_size).mean()
                ax4.plot(episodes, moving_avg, label=f'{alg.upper()}', 
                        color=colors[i], linewidth=2.5)
            else:
                ax4.plot(episodes, rewards_history, label=f'{alg.upper()}', 
                        color=colors[i], linewidth=2.5)
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Reward')
        ax4.set_title('Learning Curves (Moving Average)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.plots_dir}/performance_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Performance comparison saved to: {filename}")
        
        return filename
    
    def create_action_analysis(self):
        """Create action distribution analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Action Distribution Analysis', fontsize=16, fontweight='bold')
        
        algorithms = list(self.results_data.keys())
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICKUP', 'DROP']
        
        # 1. Action Distribution Heatmap
        ax1 = axes[0]
        action_matrix = []
        
        for alg in algorithms:
            action_dist = self.results_data[alg]['action_distribution']
            total_actions = sum(action_dist.values())
            percentages = [action_dist[i]/total_actions * 100 for i in range(6)]
            action_matrix.append(percentages)
        
        im = ax1.imshow(action_matrix, cmap='YlOrRd', aspect='auto')
        ax1.set_xticks(range(6))
        ax1.set_xticklabels(action_names, rotation=45)
        ax1.set_yticks(range(len(algorithms)))
        ax1.set_yticklabels([alg.upper() for alg in algorithms])
        ax1.set_title('Action Usage Percentage')
        
        # Add percentage text
        for i in range(len(algorithms)):
            for j in range(6):
                text = ax1.text(j, i, f'{action_matrix[i][j]:.1f}%',
                              ha="center", va="center", color="black", fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1, shrink=0.6)
        cbar.set_label('Usage Percentage (%)')
        
        # 2. Action Entropy Comparison
        ax2 = axes[1]
        entropies = []
        
        for alg in algorithms:
            action_dist = self.results_data[alg]['action_distribution']
            total = sum(action_dist.values())
            entropy = 0
            for count in action_dist.values():
                if count > 0:
                    prob = count / total
                    entropy -= prob * np.log2(prob)
            entropies.append(entropy)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax2.bar(algorithms, entropies, color=colors[:len(algorithms)], alpha=0.8)
        ax2.set_title('Action Diversity (Entropy)')
        ax2.set_ylabel('Entropy (bits)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, entropy in zip(bars, entropies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{entropy:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.plots_dir}/action_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Action analysis saved to: {filename}")
        
        return filename
    
    def create_hyperparameter_analysis(self):
        """Create hyperparameter comparison visualization"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create hyperparameter comparison table
        hyperparams_data = []
        algorithms = list(self.results_data.keys())
        
        for alg in algorithms:
            params = self.results_data[alg]['hyperparameters']
            hyperparams_data.append([
                alg.upper(),
                params.get('learning_rate', 'N/A'),
                params.get('gamma', 'N/A'),
                params.get('batch_size', params.get('n_steps', 'N/A')),
                self.results_data[alg]['mean_reward']
            ])
        
        # Create table
        columns = ['Algorithm', 'Learning Rate', 'Gamma', 'Batch/Steps', 'Performance']
        
        # Create a color map based on performance
        performances = [row[4] for row in hyperparams_data]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(hyperparams_data)))
        
        table = ax.table(cellText=hyperparams_data,
                        colLabels=columns,
                        cellLoc='center',
                        loc='center',
                        cellColours=[colors for _ in range(len(hyperparams_data))])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Hyperparameter Configuration Comparison', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add performance ranking
        sorted_algos = sorted(zip(algorithms, performances), key=lambda x: x[1], reverse=True)
        ranking_text = "Performance Ranking:\n"
        for i, (alg, perf) in enumerate(sorted_algos, 1):
            ranking_text += f"{i}. {alg.upper()}: {perf:.2f}\n"
        
        ax.text(1.1, 0.5, ranking_text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.plots_dir}/hyperparameter_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Hyperparameter analysis saved to: {filename}")
        
        return filename
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        fig = plt.figure(figsize=(16, 12))
        
        # Title
        fig.suptitle('Reinforcement Learning Algorithm Comparison Report', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        algorithms = list(self.results_data.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # 1. Overall Performance Radar Chart
        ax1 = fig.add_subplot(gs[0:2, 0:2], projection='polar')
        
        # Metrics for radar chart
        metrics = ['Mean Reward', 'Success Rate', 'Exploration', 'Stability']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        
        for i, alg in enumerate(algorithms):
            values = [
                self.results_data[alg]['mean_reward'] / 5.0,  # Normalize to 0-1
                self.results_data[alg]['success_rate'] / 100.0,
                self.results_data[alg]['mean_exploration'],
                1.0 / (1.0 + self.results_data[alg]['performance_stability'])  # Inverse stability
            ]
            values += values[:1]  # Close the circle
            
            ax1.plot(angles, values, 'o-', linewidth=2, label=alg.upper(), color=colors[i])
            ax1.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metrics)
        ax1.set_ylim(0, 1)
        ax1.set_title('Overall Performance Profile', fontweight='bold', pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 2. Key Statistics Table
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.axis('off')
        
        stats_data = []
        for alg in algorithms:
            stats_data.append([
                alg.upper(),
                f"{self.results_data[alg]['mean_reward']:.2f}",
                f"{self.results_data[alg]['success_rate']:.1f}%",
                f"{self.results_data[alg]['mean_exploration']:.3f}"
            ])
        
        table = ax2.table(cellText=stats_data,
                         colLabels=['Algorithm', 'Avg Reward', 'Success Rate', 'Exploration'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax2.set_title('Key Performance Metrics', fontweight='bold', y=0.8)
        
        # 3. Learning Progression
        ax3 = fig.add_subplot(gs[1, 2:])
        
        for i, alg in enumerate(algorithms):
            rewards = self.results_data[alg]['episode_rewards']
            episodes = range(1, len(rewards) + 1)
            
            # Calculate moving average
            window_size = max(3, len(rewards)//10)
            moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
            
            ax3.plot(episodes, moving_avg, label=alg.upper(), 
                    color=colors[i], linewidth=2)
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        ax3.set_title('Learning Progression', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Algorithm Strengths Summary
        ax4 = fig.add_subplot(gs[2:, :])
        ax4.axis('off')
        
        # Find best performer in each category
        best_reward = max(algorithms, key=lambda x: self.results_data[x]['mean_reward'])
        best_success = max(algorithms, key=lambda x: self.results_data[x]['success_rate'])
        best_exploration = max(algorithms, key=lambda x: self.results_data[x]['mean_exploration'])
        best_stability = min(algorithms, key=lambda x: self.results_data[x]['performance_stability'])
        
        summary_text = f"""
ALGORITHM ANALYSIS SUMMARY

BEST PERFORMERS:
• Highest Average Reward: {best_reward.upper()} ({self.results_data[best_reward]['mean_reward']:.2f})
• Highest Success Rate: {best_success.upper()} ({self.results_data[best_success]['success_rate']:.1f}%)
• Best Exploration: {best_exploration.upper()} ({self.results_data[best_exploration]['mean_exploration']:.3f})
• Most Stable: {best_stability.upper()} (σ = {self.results_data[best_stability]['performance_stability']:.3f})

ALGORITHM CHARACTERISTICS:

DQN (Deep Q-Network):
• Value-based method using deep neural networks
• Good for environments with discrete action spaces
• Benefits from experience replay and target networks
• Hyperparameters: Learning rate affects convergence speed

PPO (Proximal Policy Optimization):
• Policy gradient method with clipped objective
• Stable training with good sample efficiency
• Balances exploration and exploitation well
• Hyperparameters: n_steps controls policy update frequency

A2C (Advantage Actor-Critic):
• Combines value and policy learning
• Faster than PPO but potentially less stable
• Good for environments requiring quick adaptation
• Hyperparameters: n_steps affects bias-variance tradeoff

RECOMMENDATIONS:
• For maximum reward: Use {best_reward.upper()}
• For reliable performance: Use {best_success.upper()}
• For exploration tasks: Use {best_exploration.upper()}
• For stable training: Use {best_stability.upper()}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Save the comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.plots_dir}/comprehensive_report_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comprehensive report saved to: {filename}")
        
        return filename
    
    def generate_all_visualizations(self):
        """Generate all visualization plots"""
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 60)
        
        self.load_results()
        
        plots_created = []
        
        print("Creating performance comparison...")
        plots_created.append(self.create_performance_comparison())
        
        print("Creating action analysis...")
        plots_created.append(self.create_action_analysis())
        
        print("Creating hyperparameter analysis...")
        plots_created.append(self.create_hyperparameter_analysis())
        
        print("Creating comprehensive summary report...")
        plots_created.append(self.create_summary_report())
        
        print(f"\nVISUALIZATION COMPLETE!")
        print(f"Generated {len(plots_created)} visualization files:")
        for plot in plots_created:
            print(f"  {plot}")
        
        return plots_created

def main():
    """Main visualization function"""
    visualizer = RLVisualizationSystem()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()
