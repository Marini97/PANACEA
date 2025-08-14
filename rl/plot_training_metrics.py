"""
Training Metrics Plotting Utility

This module provides utilities to load and visualize training metrics
saved during Actor-Critic training for the ADT environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import Optional, Dict, List
import argparse
import os


class TrainingPlotter:
    """Utility class for plotting training metrics from saved data."""
    
    def __init__(self, metrics_file: str):
        """Initialize plotter with metrics file."""
        self.metrics_file = metrics_file
        self.metrics_data = None
        self.load_metrics()
    
    def load_metrics(self):
        """Load training metrics from file."""
        try:
            with open(self.metrics_file, 'rb') as f:
                self.metrics_data = pickle.load(f)
            print(f"Training metrics loaded from {self.metrics_file}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_file}")
        except Exception as e:
            raise RuntimeError(f"Error loading metrics: {e}")
    
    def plot_training_progress(self, save_path: Optional[str] = None, show_plot: bool = True):
        """Plot comprehensive training progress."""
        if self.metrics_data is None:
            raise RuntimeError("No metrics data loaded")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Actor-Critic Training Progress', fontsize=16)
        
        # Moving average function
        def moving_average(data, window=100):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Episode rewards
        attacker_rewards = self.metrics_data['attacker_rewards']
        defender_rewards = self.metrics_data['defender_rewards']
        
        axes[0, 0].plot(moving_average(attacker_rewards), label='Attacker', color='red', alpha=0.8)
        axes[0, 0].plot(moving_average(defender_rewards), label='Defender', color='blue', alpha=0.8)
        axes[0, 0].set_title('Episode Rewards (Moving Average)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Win rates
        win_window = 50
        attacker_wins = self.metrics_data['attacker_wins']
        defender_wins = self.metrics_data['defender_wins']
        
        att_win_rate = moving_average([np.mean(attacker_wins[max(0, i-win_window):i+1]) 
                                      for i in range(len(attacker_wins))])
        def_win_rate = moving_average([np.mean(defender_wins[max(0, i-win_window):i+1]) 
                                      for i in range(len(defender_wins))])
        
        axes[0, 1].plot(att_win_rate, label='Attacker', color='red', alpha=0.8)
        axes[0, 1].plot(def_win_rate, label='Defender', color='blue', alpha=0.8)
        axes[0, 1].set_title(f'Win Rates (Rolling {win_window}-episode window)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Episode lengths
        episode_lengths = self.metrics_data['episode_lengths']
        axes[0, 2].plot(moving_average(episode_lengths), color='green', alpha=0.8)
        axes[0, 2].set_title('Episode Lengths (Moving Average)')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Steps')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Training losses - Attacker
        attacker_losses = self.metrics_data['attacker_losses']
        if attacker_losses['total']:
            axes[1, 0].plot(moving_average(attacker_losses['actor']), 
                           label='Actor', color='orange', alpha=0.8)
            axes[1, 0].plot(moving_average(attacker_losses['critic']), 
                           label='Critic', color='purple', alpha=0.8)
            axes[1, 0].plot(moving_average(attacker_losses['total']), 
                           label='Total', color='red', alpha=0.8)
        axes[1, 0].set_title('Attacker Training Losses')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training losses - Defender
        defender_losses = self.metrics_data['defender_losses']
        if defender_losses['total']:
            axes[1, 1].plot(moving_average(defender_losses['actor']), 
                           label='Actor', color='orange', alpha=0.8)
            axes[1, 1].plot(moving_average(defender_losses['critic']), 
                           label='Critic', color='purple', alpha=0.8)
            axes[1, 1].plot(moving_average(defender_losses['total']), 
                           label='Total', color='blue', alpha=0.8)
        axes[1, 1].set_title('Defender Training Losses')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Final performance distribution
        if len(attacker_rewards) > 0:
            recent_window = min(200, len(attacker_rewards))
            recent_att = attacker_rewards[-recent_window:]
            recent_def = defender_rewards[-recent_window:]

            range_min = min(min(recent_att), min(recent_def))
            range_max = max(max(recent_att), max(recent_def))

            axes[1, 2].hist(recent_att, range=(range_min, range_max), alpha=0.6, label='Attacker', color='red')
            axes[1, 2].hist(recent_def, range=(range_min, range_max), alpha=0.6, label='Defender', color='blue')
            axes[1, 2].set_title(f'Recent Reward Distribution (Last {recent_window} episodes)')
            axes[1, 2].set_xlabel('Reward')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_epsilon_decay(self, save_path: Optional[str] = None, show_plot: bool = True):
        """Plot epsilon decay over time if available."""
        if self.metrics_data is None:
            raise RuntimeError("No metrics data loaded")
        
        attacker_epsilon = self.metrics_data.get('attacker_epsilon', [])
        defender_epsilon = self.metrics_data.get('defender_epsilon', [])
        
        if not attacker_epsilon and not defender_epsilon:
            print("No epsilon data available to plot")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        if attacker_epsilon:
            ax.plot(attacker_epsilon, label='Attacker', color='red', alpha=0.8)
        if defender_epsilon:
            ax.plot(defender_epsilon, label='Defender', color='blue', alpha=0.8)
        
        ax.set_title('Epsilon Decay During Training')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_reward_comparison(self, save_path: Optional[str] = None, show_plot: bool = True):
        """Plot detailed reward comparison between agents."""
        if self.metrics_data is None:
            raise RuntimeError("No metrics data loaded")
        
        attacker_rewards = self.metrics_data['attacker_rewards']
        defender_rewards = self.metrics_data['defender_rewards']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Reward Analysis', fontsize=16)
        
        # Moving average function
        def moving_average(data, window=100):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Raw rewards
        axes[0, 0].plot(attacker_rewards, label='Attacker', color='red', alpha=0.5)
        axes[0, 0].plot(defender_rewards, label='Defender', color='blue', alpha=0.5)
        axes[0, 0].set_title('Raw Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Smoothed rewards
        axes[0, 1].plot(moving_average(attacker_rewards), label='Attacker', color='red', alpha=0.8)
        axes[0, 1].plot(moving_average(defender_rewards), label='Defender', color='blue', alpha=0.8)
        axes[0, 1].set_title('Smoothed Episode Rewards (100-episode MA)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative rewards
        axes[1, 0].plot(np.cumsum(attacker_rewards), label='Attacker', color='red', alpha=0.8)
        axes[1, 0].plot(np.cumsum(defender_rewards), label='Defender', color='blue', alpha=0.8)
        axes[1, 0].set_title('Cumulative Rewards')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Cumulative Reward')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Reward difference
        reward_diff = np.array(attacker_rewards) - np.array(defender_rewards)
        axes[1, 1].plot(moving_average(reward_diff), color='green', alpha=0.8)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Reward Difference (Attacker - Defender)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward Difference')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def print_training_summary(self):
        """Print a summary of training metrics."""
        if self.metrics_data is None:
            raise RuntimeError("No metrics data loaded")
        
        attacker_rewards = self.metrics_data['attacker_rewards']
        defender_rewards = self.metrics_data['defender_rewards']
        episode_lengths = self.metrics_data['episode_lengths']
        attacker_wins = self.metrics_data['attacker_wins']
        defender_wins = self.metrics_data['defender_wins']
        
        print("=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total Episodes: {len(attacker_rewards)}")
        print(f"Average Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        print()
        print("REWARDS:")
        print(f"  Attacker - Mean: {np.mean(attacker_rewards):.3f}, Std: {np.std(attacker_rewards):.3f}")
        print(f"  Defender - Mean: {np.mean(defender_rewards):.3f}, Std: {np.std(defender_rewards):.3f}")
        print()
        print("WIN RATES:")
        print(f"  Attacker: {np.mean(attacker_wins):.1%}")
        print(f"  Defender: {np.mean(defender_wins):.1%}")
        print()
        
        # Recent performance (last 100 episodes)
        if len(attacker_rewards) >= 100:
            recent_att = attacker_rewards[-100:]
            recent_def = defender_rewards[-100:]
            recent_att_wins = attacker_wins[-100:]
            recent_def_wins = defender_wins[-100:]
            
            print("RECENT PERFORMANCE (Last 100 episodes):")
            print(f"  Attacker - Mean Reward: {np.mean(recent_att):.3f}")
            print(f"  Defender - Mean Reward: {np.mean(recent_def):.3f}")
            print(f"  Attacker Win Rate: {np.mean(recent_att_wins):.1%}")
            print(f"  Defender Win Rate: {np.mean(recent_def_wins):.1%}")
        
        print("=" * 60)


def main():
    """Command-line interface for plotting training metrics."""
    parser = argparse.ArgumentParser(description='Plot training metrics from saved data')
    parser.add_argument('metrics_file', help='Path to the metrics pickle file')
    parser.add_argument('--save-dir', '-s', help='Directory to save plots')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display plots on screen')
    parser.add_argument('--summary', action='store_true', help='Print training summary')
    parser.add_argument('--epsilon', action='store_true', help='Plot epsilon decay')
    parser.add_argument('--rewards', action='store_true', help='Plot detailed reward comparison')
    
    args = parser.parse_args()
    
    # Check if metrics file exists
    if not os.path.exists(args.metrics_file):
        print(f"Error: Metrics file not found: {args.metrics_file}")
        return
    
    # Create plotter
    plotter = TrainingPlotter(args.metrics_file)
    
    # Print summary if requested
    if args.summary:
        plotter.print_training_summary()
    
    show_plots = not args.no_show
    
    # Generate save paths
    base_name = os.path.splitext(os.path.basename(args.metrics_file))[0]
    save_dir = args.save_dir or os.path.dirname(args.metrics_file)
    
    # Plot main training progress (default)
    if not any([args.epsilon, args.rewards]):
        save_path = os.path.join(save_dir, f"{base_name}_training_progress.png") if args.save_dir else None
        plotter.plot_training_progress(save_path=save_path, show_plot=show_plots)
    
    # Plot epsilon decay if requested
    if args.epsilon:
        save_path = os.path.join(save_dir, f"{base_name}_epsilon_decay.png") if args.save_dir else None
        plotter.plot_epsilon_decay(save_path=save_path, show_plot=show_plots)
    
    # Plot detailed rewards if requested
    if args.rewards:
        save_path = os.path.join(save_dir, f"{base_name}_reward_comparison.png") if args.save_dir else None
        plotter.plot_reward_comparison(save_path=save_path, show_plot=show_plots)


if __name__ == "__main__":
    main()
