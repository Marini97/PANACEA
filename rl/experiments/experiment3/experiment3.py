# Import Required Libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate to the parent directory that contains core
parent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
# Add to Python path
sys.path.append(parent_dir)

# Now the imports should work
from core.adt_trainer import ADTTrainer
from core.adt_actor_critic import ADTAgent, TrainingMetrics
from core.adt_env import AttackDefenseTreeMultiAgentEnv
from core.adt_time_env import AttackDefenseTreeMultiAgentTimeEnv

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# set device
device_name = "cpu" # Force CPU usage for small models
device = torch.device(device_name)


class Experiment3Trainer(ADTTrainer):
    """
    Specialized trainer for Experiment 3 that can work with both basic and time-based ADT environments.
    """
    
    def __init__(self, config, env_type='basic'):
        super().__init__(config)
        self.env_type = env_type
    
    def get_environment_class(self):
        """Get the appropriate environment class based on environment type."""
        if self.env_type == 'time':
            return AttackDefenseTreeMultiAgentTimeEnv
        elif self.env_type == 'basic':
            return AttackDefenseTreeMultiAgentEnv
        else:
            raise ValueError(f"Unknown environment type: {self.env_type}")


class Experiment3Runner:
    """
    Main class to orchestrate all training runs for Experiment 3.
    """
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.results_dir = Path(script_dir) / "results"
        self.envs_dir = Path(script_dir) / "envs"
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(exist_ok=True)
        
        # Store all results for averaging
        self.all_results = {}
    
    def get_env_files(self) -> List[str]:
        """Get all environment JSON files from the envs directory."""
        env_files = list(self.envs_dir.glob("*.json"))
        return [str(f) for f in env_files]
    
    def create_config_for_run(self, env_file: str, env_type: str, run_id: int, max_episodes: int) -> Dict[str, Any]:
        """Create configuration for a specific training run."""
        config = self.base_config.copy()
        
        # Update specific parameters for this run
        config.update({
            'env_file': env_file,
            'max_episodes': max_episodes,
            'seed': self.base_config['seed'] + run_id,  # Different seed for each run
            'device': device_name
        })
        
        return config
    
    def run_single_training(self, env_file: str, env_type: str, run_id: int, max_episodes: int) -> Dict[str, Any] | None:
        """Run a single training session and return results."""
        env_name = Path(env_file).stem
        
        print(f"\n{'='*80}")
        print(f"🚀 Starting Training: {env_name} | Type: {env_type} | Run: {run_id+1} | Episodes: {max_episodes}")
        print(f"{'='*80}")
        
        # Create configuration for this run
        config = self.create_config_for_run(env_file, env_type, run_id, max_episodes)
        
        if env_type == 'time':
            config['log_dir'] = f"{config['log_dir']}/{env_name}/time/episodes_{max_episodes}/run_{run_id}/training_logs"
        else:
            config['log_dir'] = f"{config['log_dir']}/{env_name}/basic/episodes_{max_episodes}/run_{run_id}/training_logs"

        # Create trainer
        trainer = Experiment3Trainer(config, env_type)
        
        # Get the appropriate environment class
        env_class = trainer.get_environment_class()
        
        try:
            # Train the agents
            start_time = time.time()
            env, attacker_agent, defender_agent, metrics = trainer.train(env_class, env_file)
            training_time = time.time() - start_time
            
            # Extract final training metrics
            training_results = {
                'attacker_rewards': metrics.attacker_rewards.copy(),
                'defender_rewards': metrics.defender_rewards.copy(),
                'episode_lengths': metrics.episode_lengths.copy(),
                'attacker_wins': metrics.attacker_wins.copy(),
                'defender_wins': metrics.defender_wins.copy(),
                'training_time': training_time,
                'config': config,
                'env_type': env_type,
                'run_id': run_id,
                'max_episodes': max_episodes
            }
            
            # Save individual run results
            self.save_run_results(env_name, env_type, max_episodes, run_id, training_results)

            print(f"✅ Training completed successfully!")
            print(f"   Training time: {training_time/60:.2f} minutes")
            print(f"   Final attacker reward: {training_results['attacker_rewards'][-1]:.4f}")
            print(f"   Final defender reward: {training_results['defender_rewards'][-1]:.4f}")
            
            return training_results
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            return None

    def save_run_results(self, env_name: str, env_type: str, max_episodes: int, run_id: int, results: Dict[str, Any]):
        """Save results for a single run."""
        results_path = self.results_dir / env_name / env_type / f"episodes_{max_episodes}/run_{run_id}"
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete results as JSON
        results_file = results_path / "training_results.json"
        
        # Prepare serializable results
        serializable_results = results.copy()
        serializable_results['attacker_rewards'] = [float(x) for x in results['attacker_rewards']]
        serializable_results['defender_rewards'] = [float(x) for x in results['defender_rewards']]
        serializable_results['episode_lengths'] = [int(x) for x in results['episode_lengths']]
        serializable_results['attacker_wins'] = [int(x) for x in results['attacker_wins']]
        serializable_results['defender_wins'] = [int(x) for x in results['defender_wins']]
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
    
    def run_all_experiments(self, num_runs: int = 5, episode_ranges: List[int] | None = None):
        """
        Run all experiments for all environment files.
        
        Args:
            num_runs: Number of training runs per configuration
            episode_ranges: List of episode counts to test (e.g., [1000, 2000, 3000, 5000])
        """
        if episode_ranges is None:
            episode_ranges = [1000, 2000, 3000, 5000]
        
        # Get all environment files
        env_files = self.get_env_files()
        
        if not env_files:
            print("❌ No environment files found in 'envs' directory!")
            return
        
        print(f"🔍 Found {len(env_files)} environment files:")
        for env_file in env_files:
            print(f"   - {env_file}")
        
        total_experiments = len(env_files) * 2 * len(episode_ranges) * num_runs  # 2 for basic and time
        
        print(f"\n📊 Total experiments to run: {total_experiments}")
        print(f"   - Environment files: {len(env_files)}")
        print(f"   - Environment types: 2 (basic, time)")
        print(f"   - Episode ranges: {episode_ranges}")
        print(f"   - Runs per configuration: {num_runs}")
        
        # Create overall progress bar
        with tqdm(total=total_experiments, desc="Overall Progress", unit="experiment") as pbar:
            for env_file in env_files:
                env_name = Path(env_file).stem
                
                for env_type in ['basic', 'time']:
                    for max_episodes in episode_ranges:
                        
                        print(f"\n{'#'*100}")
                        print(f"🎯 EXPERIMENT GROUP: {env_name} | {env_type} | {max_episodes} episodes")
                        print(f"{'#'*100}")
                        
                        # Store results for this configuration
                        config_key = f"{env_name}_{env_type}_{max_episodes}"
                        self.all_results[config_key] = []
                        
                        # Progress bar for runs within this configuration
                        for run_id in tqdm(range(num_runs), 
                                         desc=f"{env_name}_{env_type}_{max_episodes}", 
                                         unit="run", 
                                         leave=False):
                            
                            # Run single training
                            result = self.run_single_training(env_file, env_type, run_id, max_episodes)
                            
                            if result is not None:
                                self.all_results[config_key].append(result)
                            
                            # Update overall progress
                            pbar.update(1)
                            pbar.set_postfix({
                                'env': env_name,
                                'type': env_type,
                                'episodes': max_episodes,
                                'run': run_id + 1
                            })
                            
                            # Small delay between runs
                            time.sleep(1)
                        
                        # Generate averaged results for this configuration
                        self.generate_averaged_results(env_name, env_type, max_episodes)
        
        # Generate final comprehensive report
        self.generate_final_report()
        
        print(f"\n🎉 ALL EXPERIMENTS COMPLETED!")
        print(f"📁 Results saved in: {self.results_dir}")
    
    def generate_averaged_results(self, env_name: str, env_type: str, max_episodes: int):
        """Generate averaged results for a specific configuration."""
        config_key = f"{env_name}_{env_type}_{max_episodes}"
        
        if config_key not in self.all_results or not self.all_results[config_key]:
            print(f"⚠️ No results found for {config_key}")
            return
        
        results = self.all_results[config_key]
        num_runs = len(results)
        
        print(f"\n📊 Generating averaged results for {config_key} ({num_runs} runs)")
        
        # Average the training curves
        max_length = max(len(result['attacker_rewards']) for result in results)
        
        # Pad shorter sequences with their last value
        padded_attacker_rewards = []
        padded_defender_rewards = []
        padded_episode_lengths = []
        
        for result in results:
            att_rewards = result['attacker_rewards']
            def_rewards = result['defender_rewards']
            ep_lengths = result['episode_lengths']
            
            # Pad with last value if shorter
            if len(att_rewards) < max_length:
                last_att = att_rewards[-1] if att_rewards else 0
                last_def = def_rewards[-1] if def_rewards else 0
                last_len = ep_lengths[-1] if ep_lengths else 0
                
                att_rewards = att_rewards + [last_att] * (max_length - len(att_rewards))
                def_rewards = def_rewards + [last_def] * (max_length - len(def_rewards))
                ep_lengths = ep_lengths + [last_len] * (max_length - len(ep_lengths))
            
            padded_attacker_rewards.append(att_rewards)
            padded_defender_rewards.append(def_rewards)
            padded_episode_lengths.append(ep_lengths)
        
        # Calculate averages and standard deviations
        avg_attacker_rewards = np.mean(padded_attacker_rewards, axis=0)
        std_attacker_rewards = np.std(padded_attacker_rewards, axis=0)
        
        avg_defender_rewards = np.mean(padded_defender_rewards, axis=0)
        std_defender_rewards = np.std(padded_defender_rewards, axis=0)
        
        avg_episode_lengths = np.mean(padded_episode_lengths, axis=0)
        std_episode_lengths = np.std(padded_episode_lengths, axis=0)
        
        # Calculate final metrics
        final_attacker_rewards = [result['attacker_rewards'][-1] for result in results if result['attacker_rewards']]
        final_defender_rewards = [result['defender_rewards'][-1] for result in results if result['defender_rewards']]
        training_times = [result['training_time'] for result in results]
        
        averaged_results = {
            'config_key': config_key,
            'env_name': env_name,
            'env_type': env_type,
            'max_episodes': max_episodes,
            'num_runs': num_runs,
            'avg_attacker_rewards': avg_attacker_rewards.tolist(),
            'std_attacker_rewards': std_attacker_rewards.tolist(),
            'avg_defender_rewards': avg_defender_rewards.tolist(),
            'std_defender_rewards': std_defender_rewards.tolist(),
            'avg_episode_lengths': avg_episode_lengths.tolist(),
            'std_episode_lengths': std_episode_lengths.tolist(),
            'final_metrics': {
                'avg_final_attacker_reward': np.mean(final_attacker_rewards),
                'std_final_attacker_reward': np.std(final_attacker_rewards),
                'avg_final_defender_reward': np.mean(final_defender_rewards),
                'std_final_defender_reward': np.std(final_defender_rewards),
                'avg_training_time': np.mean(training_times),
                'std_training_time': np.std(training_times),
            },
            'individual_runs': results
        }
        
        # Save averaged results
        avg_results_path = self.results_dir / env_name / env_type / f"episodes_{max_episodes}"
        avg_results_path.mkdir(parents=True, exist_ok=True)
        
        avg_results_file = avg_results_path / "averaged_results.json"
        with open(avg_results_file, 'w') as f:
            json.dump(averaged_results, f, indent=2, default=str)
        
        print(f"📈 Averaged results saved to: {avg_results_file}")
        print(f"   Final attacker reward: {averaged_results['final_metrics']['avg_final_attacker_reward']:.4f} ± {averaged_results['final_metrics']['std_final_attacker_reward']:.4f}")
        print(f"   Final defender reward: {averaged_results['final_metrics']['avg_final_defender_reward']:.4f} ± {averaged_results['final_metrics']['std_final_defender_reward']:.4f}")
        print(f"   Average training time: {averaged_results['final_metrics']['avg_training_time']/60:.2f} ± {averaged_results['final_metrics']['std_training_time']/60:.2f} minutes")
    
    def generate_final_report(self):
        """Generate a comprehensive final report of all experiments."""
        print(f"\n📋 Generating final comprehensive report...")
        
        print(f"\n🎉 Final report generated!")
        print(f"📁 Results saved in: {self.results_dir}")
        print(f"📊 Use 'python plotting.py' to regenerate all plots")


def create_base_config(results_dir: Path) -> Dict[str, Any]:
    """Create base configuration for all experiments."""
    return {
        # Training hyperparameters
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'hidden_size': 128,
        'max_steps': 100,
        
        # Epsilon exploration parameters
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 5000,
        
        # Training control
        'update_frequency': 10,
        'print_frequency': 10000,
        
        # System settings
        'device': device_name,
        'seed': 42,
        
        # Directory settings
        'model_dir': f'{results_dir}',
        'log_dir': f'{results_dir}',
        'save_models': False,
        'save_logs': True,
        
        # Testing
        'test_games': 0,
    }


def main():
    print("🚀 Starting Experiment 3: Sequential Training Analysis")
    print("="*80)
    
    # Define results directory
    results_dir = Path(script_dir) / "results"
    
    # Create base configuration
    base_config = create_base_config(results_dir)
    
    # Create experiment runner
    runner = Experiment3Runner(base_config)
    
    # Configuration for the experiments
    num_runs = 5  # Number of training runs per configuration
    episode_ranges = [3000, 5000, 10000]  # Different episode counts to test

    print(f"📋 Experiment Configuration:")
    print(f"   Number of runs per configuration: {num_runs}")
    print(f"   Episode ranges to test: {episode_ranges}")
    print(f"   Environment types: basic, time")
    print(f"   Device: {device_name}")
    print("="*80)
    
    # Run all experiments
    runner.run_all_experiments(num_runs=num_runs, episode_ranges=episode_ranges)


if __name__ == "__main__":
    main()




