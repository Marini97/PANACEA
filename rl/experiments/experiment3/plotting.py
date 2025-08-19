"""
Plotting utilities for Experiment 3 results.
This module provides functions to generate individual plots from saved experiment results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any
import glob

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Experiment3Plotter:
    """
    Plotting utilities for Experiment 3 results.
    Generates individual plots from saved experiment data.
    """
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        
    def create_plots(self, averaged_results: Dict[str, Any], save_path: Path):
        """Create individual visualization plots for averaged results."""
        config_key = averaged_results['config_key']
        episodes = range(len(averaged_results['avg_attacker_rewards']))
        
        # Create plots directory
        plots_dir = save_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        saved_plots = []
        
        # Calculate rolling averages (window of 100 episodes)
        window_size = min(100, len(averaged_results['avg_attacker_rewards']) // 10)
        if window_size < 10:
            window_size = 10
        
        # Rolling average for attacker rewards
        att_rewards = np.array(averaged_results['avg_attacker_rewards'])
        att_rolling = np.convolve(att_rewards, np.ones(window_size)/window_size, mode='valid')
        att_rolling_episodes = episodes[window_size-1:]
        
        # Rolling average for defender rewards
        def_rewards = np.array(averaged_results['avg_defender_rewards'])
        def_rolling = np.convolve(def_rewards, np.ones(window_size)/window_size, mode='valid')
        
        # Plot 1: Attacker Rewards with Rolling Average
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, averaged_results['avg_attacker_rewards'], label='Episode Rewards', color='lightcoral', alpha=0.6, linewidth=1)
        plt.plot(att_rolling_episodes, att_rolling, label=f'Rolling Average (window={window_size})', color='red', linewidth=3)
        plt.fill_between(episodes, 
                        np.array(averaged_results['avg_attacker_rewards']) - np.array(averaged_results['std_attacker_rewards']),
                        np.array(averaged_results['avg_attacker_rewards']) + np.array(averaged_results['std_attacker_rewards']),
                        alpha=0.2, color='red', label='±1 STD')
        plt.title(f'Attacker Rewards - {config_key}', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        plot_file = plots_dir / f"{config_key}_attacker_rewards.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(plot_file)
        
        # Plot 2: Defender Rewards with Rolling Average
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, averaged_results['avg_defender_rewards'], label='Episode Rewards', color='lightblue', alpha=0.6, linewidth=1)
        plt.plot(att_rolling_episodes, def_rolling, label=f'Rolling Average (window={window_size})', color='blue', linewidth=3)
        plt.fill_between(episodes,
                        np.array(averaged_results['avg_defender_rewards']) - np.array(averaged_results['std_defender_rewards']),
                        np.array(averaged_results['avg_defender_rewards']) + np.array(averaged_results['std_defender_rewards']),
                        alpha=0.2, color='blue', label='±1 STD')
        plt.title(f'Defender Rewards - {config_key}', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        plot_file = plots_dir / f"{config_key}_defender_rewards.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(plot_file)
        
        # Plot 3: Win Rate Analysis
        if 'avg_attacker_wins' in averaged_results and 'avg_defender_wins' in averaged_results:
            att_wins = np.array(averaged_results['avg_attacker_wins'])
            def_wins = np.array(averaged_results['avg_defender_wins'])
        else:
            # Calculate win rates from individual runs if available
            att_wins = []
            def_wins = []
            if 'individual_runs' in averaged_results:
                for episode_idx in range(len(episodes)):
                    episode_att_wins = []
                    episode_def_wins = []
                    for run in averaged_results['individual_runs']:
                        if episode_idx < len(run['attacker_wins']):
                            episode_att_wins.append(run['attacker_wins'][episode_idx])
                            episode_def_wins.append(run['defender_wins'][episode_idx])
                    att_wins.append(np.mean(episode_att_wins) if episode_att_wins else 0)
                    def_wins.append(np.mean(episode_def_wins) if episode_def_wins else 0)
            else:
                # Fallback: assume equal win rates
                att_wins = [0.5] * len(episodes)
                def_wins = [0.5] * len(episodes)
        
        # Rolling average for win rates
        att_winrate_rolling = np.convolve(att_wins, np.ones(window_size)/window_size, mode='valid')
        def_winrate_rolling = np.convolve(def_wins, np.ones(window_size)/window_size, mode='valid')
        
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, att_wins, label='Attacker Win Rate (Episode)', color='lightcoral', alpha=0.6, linewidth=1)
        plt.plot(episodes, def_wins, label='Defender Win Rate (Episode)', color='lightblue', alpha=0.6, linewidth=1)
        plt.plot(att_rolling_episodes, att_winrate_rolling, label=f'Attacker Rolling Avg (window={window_size})', color='red', linewidth=3)
        plt.plot(att_rolling_episodes, def_winrate_rolling, label=f'Defender Rolling Avg (window={window_size})', color='blue', linewidth=3)
        plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Equal Win Rate')
        plt.title(f'Win Rates Over Training - {config_key}', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Win Rate', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        plot_file = plots_dir / f"{config_key}_win_rates.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(plot_file)
        
        # Plot 4: Episode Lengths
        episode_lengths = np.array(averaged_results['avg_episode_lengths'])
        lengths_rolling = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
        
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, averaged_results['avg_episode_lengths'], label='Episode Lengths', color='lightgreen', alpha=0.6, linewidth=1)
        plt.plot(att_rolling_episodes, lengths_rolling, label=f'Rolling Average (window={window_size})', color='green', linewidth=3)
        plt.fill_between(episodes,
                        np.array(averaged_results['avg_episode_lengths']) - np.array(averaged_results['std_episode_lengths']),
                        np.array(averaged_results['avg_episode_lengths']) + np.array(averaged_results['std_episode_lengths']),
                        alpha=0.2, color='green', label='±1 STD')
        plt.title(f'Episode Lengths - {config_key}', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Steps', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        plot_file = plots_dir / f"{config_key}_episode_lengths.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(plot_file)
        
        # Plot 5: Combined Rewards with Rolling Averages
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, averaged_results['avg_attacker_rewards'], label='Attacker (Episode)', color='lightcoral', alpha=0.4, linewidth=1)
        plt.plot(episodes, averaged_results['avg_defender_rewards'], label='Defender (Episode)', color='lightblue', alpha=0.4, linewidth=1)
        plt.plot(att_rolling_episodes, att_rolling, label='Attacker (Rolling Avg)', color='red', linewidth=3)
        plt.plot(att_rolling_episodes, def_rolling, label='Defender (Rolling Avg)', color='blue', linewidth=3)
        plt.title(f'Combined Rewards Comparison - {config_key}', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        plot_file = plots_dir / f"{config_key}_combined_rewards.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(plot_file)
        
        print(f"📊 Individual plots saved to: {plots_dir}")
        for plot in saved_plots:
            print(f"   - {plot.name}")
        
        return saved_plots
    
    def create_comparison_plots(self, df: pd.DataFrame, save_path: Path):
        """Create individual comparison plots across all experiments using histograms."""
        plots_dir = save_path / "comparison_plots"
        plots_dir.mkdir(exist_ok=True)
        
        saved_plots = []
        
        # Get unique episode counts and environment types for other plots
        episode_counts = sorted(df['max_episodes'].unique())
        env_types = df['env_type'].unique()
        
        # Plot 1: Training Time vs Episodes (Consolidated)
        plt.figure(figsize=(12, 8))
        
        # Create color palette and markers
        env_names = sorted(df['env_name'].unique())
        env_types = sorted(df['env_type'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(env_names)))
        line_styles = ['-', '--']
        markers = ['o', 's']
        
        for env_idx, env_name in enumerate(env_names):
            for type_idx, env_type in enumerate(env_types):
                # Get data for this environment and type combination
                subset = df[(df['env_name'] == env_name) & (df['env_type'] == env_type)]
                if not subset.empty:
                    # Sort by episodes for proper line plotting
                    subset = subset.sort_values('max_episodes')
                    episodes = subset['max_episodes'].values
                    times = subset['avg_training_time_seconds'].values / 60  # Convert to minutes
                    errors = subset['std_training_time_seconds'].values / 60
                    
                    # Plot line with error bars
                    label = f'{env_name} ({env_type})'
                    plt.errorbar(episodes, times, yerr=errors, 
                               label=label, 
                               color=colors[env_idx], 
                               linestyle=line_styles[type_idx],
                               linewidth=2.5, 
                               marker=markers[type_idx], 
                               markersize=8, 
                               capsize=4, 
                               capthick=1.5,
                               alpha=0.9)
        
        plt.title('Average Training Time vs Number of Episodes', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Episodes', fontsize=14)
        plt.ylabel('Average Training Time (minutes)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, framealpha=0.9)
        plt.tight_layout()
        
        plot_file = plots_dir / "training_time_vs_episodes.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(plot_file)
        
        # Plot 2: Performance Gap Analysis (Attacker vs Defender)
        plt.figure(figsize=(14, 10))
        
        # Calculate reward differences
        df_copy = df.copy()
        df_copy['reward_difference'] = df_copy['avg_final_attacker_reward'] - df_copy['avg_final_defender_reward']
        df_copy['performance_advantage'] = df_copy['reward_difference'].apply(
            lambda x: 'Attacker' if x > 0 else 'Defender' if x < 0 else 'Balanced'
        )
        
        # Subplot for each episode count
        for i, episodes in enumerate(episode_counts):
            plt.subplot(2, len(episode_counts), i + 1)
            
            subset = df_copy[df_copy['max_episodes'] == episodes]
            
            # Create grouped bar chart
            width = 0.35
            x_pos = np.arange(len(subset))
            
            att_bars = plt.bar(x_pos - width/2, subset['avg_final_attacker_reward'], 
                             width, label='Attacker', alpha=0.8, color='red',
                             yerr=subset['std_final_attacker_reward'], capsize=3)
            
            def_bars = plt.bar(x_pos + width/2, subset['avg_final_defender_reward'], 
                             width, label='Defender', alpha=0.8, color='blue',
                             yerr=subset['std_final_defender_reward'], capsize=3)
            
            plt.title(f'Performance Comparison\n{episodes} Episodes', fontsize=12, fontweight='bold')
            plt.ylabel('Final Reward', fontsize=10)
            plt.xticks(x_pos, [f"{row['env_name']}\n{row['env_type']}" for _, row in subset.iterrows()], 
                      fontsize=8, rotation=45)
            plt.grid(True, alpha=0.3)
            if i == 0:
                plt.legend(fontsize=9)
        
        plt.tight_layout()
        plot_file = plots_dir / "performance_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(plot_file)
        
        # Plot 3: Summary Statistics Box Plot
        plt.figure(figsize=(16, 8))
        
        # Melt the dataframe for easier plotting
        metrics = ['avg_final_attacker_reward', 'avg_final_defender_reward', 'avg_training_time_seconds']
        metric_names = ['Attacker Reward', 'Defender Reward', 'Training Time (sec)']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            plt.subplot(1, 3, i + 1)
            
            # Create box plot data
            data_for_plot = []
            labels_for_plot = []
            
            for env_type in env_types:
                for episodes in episode_counts:
                    subset = df[(df['env_type'] == env_type) & (df['max_episodes'] == episodes)]
                    if not subset.empty:
                        data_for_plot.append(subset[metric].values)
                        labels_for_plot.append(f'{env_type}\n{episodes}ep')
            
            plt.boxplot(data_for_plot, labels=labels_for_plot, patch_artist=True)
            plt.title(f'{name} Distribution', fontsize=14, fontweight='bold')
            plt.ylabel(name, fontsize=12)
            plt.xticks(rotation=45, fontsize=10)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = plots_dir / "summary_statistics_boxplot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(plot_file)
        
        print(f"📊 Comparison plots saved to: {plots_dir}")
        for plot in saved_plots:
            print(f"   - {plot.name}")
        
        return saved_plots
    
    def load_averaged_results(self, results_path: Path) -> Dict[str, Any]:
        """Load averaged results from JSON file."""
        results_file = results_path / "averaged_results.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Averaged results file not found: {results_file}")
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def find_all_averaged_results(self) -> List[Path]:
        """Find all averaged results files in the results directory."""
        pattern = str(self.results_dir / "*" / "*" / "episodes_*" / "averaged_results.json")
        result_files = glob.glob(pattern)
        return [Path(f).parent for f in result_files]
    
    def regenerate_all_plots(self):
        """Regenerate all plots from existing averaged results."""
        print("🔄 Regenerating all plots from saved results...")
        
        # Find all averaged results
        result_dirs = self.find_all_averaged_results()
        
        if not result_dirs:
            print("❌ No averaged results found!")
            return
        
        print(f"📊 Found {len(result_dirs)} configurations with results:")
        
        for result_dir in result_dirs:
            try:
                # Load averaged results
                averaged_results = self.load_averaged_results(result_dir)
                config_key = averaged_results['config_key']
                
                print(f"   🎨 Generating plots for: {config_key}")
                
                # Create individual plots
                self.create_plots(averaged_results, result_dir)
                
            except Exception as e:
                print(f"   ❌ Failed to generate plots for {result_dir}: {e}")
        
        # Generate comparison plots
        self.generate_comparison_plots()
        
        print("✅ All plots regenerated successfully!")
    
    def generate_comparison_plots(self):
        """Generate comparison plots from all available results."""
        print("📊 Generating comparison plots...")
        
        # Collect all results data
        summary_data = []
        result_dirs = self.find_all_averaged_results()
        
        for result_dir in result_dirs:
            try:
                averaged_results = self.load_averaged_results(result_dir)
                
                # Extract summary data
                config_key = averaged_results['config_key']
                parts = config_key.split('_')
                env_name = parts[0]
                env_type = parts[1]
                max_episodes = int(parts[2])
                
                summary_data.append({
                    'env_name': env_name,
                    'env_type': env_type,
                    'max_episodes': max_episodes,
                    'num_runs': averaged_results['num_runs'],
                    'avg_final_attacker_reward': averaged_results['final_metrics']['avg_final_attacker_reward'],
                    'std_final_attacker_reward': averaged_results['final_metrics']['std_final_attacker_reward'],
                    'avg_final_defender_reward': averaged_results['final_metrics']['avg_final_defender_reward'],
                    'std_final_defender_reward': averaged_results['final_metrics']['std_final_defender_reward'],
                    'avg_training_time_seconds': averaged_results['final_metrics']['avg_training_time'],
                    'std_training_time_seconds': averaged_results['final_metrics']['std_training_time'],
                })
                
            except Exception as e:
                print(f"   ⚠️ Skipping {result_dir}: {e}")
        
        if not summary_data:
            print("❌ No valid summary data found!")
            return
        
        # Create DataFrame
        df = pd.DataFrame(summary_data)
        
        # Save summary report
        report_file = self.results_dir / "experiment3_summary_report.csv"
        df.to_csv(report_file, index=False)
        
        # Create comparison plots
        self.create_comparison_plots(df, self.results_dir)
        
        print(f"📁 Summary report saved to: {report_file}")
    
    def plot_specific_config(self, env_name: str, env_type: str, max_episodes: int):
        """Generate plots for a specific configuration."""
        config_path = self.results_dir / env_name / env_type / f"episodes_{max_episodes}"
        
        if not config_path.exists():
            print(f"❌ Configuration not found: {config_path}")
            return
        
        try:
            averaged_results = self.load_averaged_results(config_path)
            config_key = averaged_results['config_key']
            
            print(f"🎨 Generating plots for specific config: {config_key}")
            self.create_plots(averaged_results, config_path)
            
        except Exception as e:
            print(f"❌ Failed to generate plots: {e}")


def main():
    """Main function for standalone plotting."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate plots for Experiment 3 results")
    parser.add_argument("--results-dir", type=str, default="results", 
                       help="Directory containing experiment results")
    parser.add_argument("--env-name", type=str, help="Specific environment name to plot")
    parser.add_argument("--env-type", type=str, choices=['basic', 'time'], 
                       help="Specific environment type to plot")
    parser.add_argument("--episodes", type=int, help="Specific episode count to plot")
    
    args = parser.parse_args()
    
    # Get script directory and results directory
    script_dir = Path(__file__).parent
    results_dir = script_dir / args.results_dir
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Create plotter
    plotter = Experiment3Plotter(results_dir)
    
    if args.env_name and args.env_type and args.episodes:
        plotter.plot_specific_config(args.env_name, args.env_type, args.episodes)
    else:
        plotter.regenerate_all_plots()


if __name__ == "__main__":
    main()
