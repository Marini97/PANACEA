#!/usr/bin/env python3
"""
Abstract Training Module for Attack-Defense Tree Environments
with Actor-Critic agents and Epsilon-Greedy Exploration

This module provides a flexible training pipeline that can work with different
ADT environment implementations (with or without time mechanics).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import argparse
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod

# Import our custom modules
from .adt_actor_critic import ADTAgent, TrainingMetrics

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ADTTrainer:
    """
    Abstract trainer for Attack-Defense Tree environments.
    
    This class provides a flexible training pipeline that can work with different
    ADT environment implementations while maintaining consistent training logic.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.env = None
        self.attacker_agent = None
        self.defender_agent = None
        self.metrics = None
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def setup_environment(self, env_class, env_spec_file: str):
        """
        Setup the environment using the provided environment class.
        
        Args:
            env_class: The environment class to instantiate
            env_spec_file: Path to the environment specification file
        """
        print("🌍 Setting up ADT Multi-Agent Environment...")
        
        # Create environment
        self.env = env_class(env_spec_file, render_mode=None)
        print("✅ Environment created successfully!")
        
        # Environment information
        print(f"\n📊 Environment Information:")
        print(f"Agents: {self.env.possible_agents}")
        print(f"Action spaces: {self.env.action_spaces}")
        print(f"Observation spaces: {self.env.observation_spaces}")
        
        # Reset to get initial observations
        obs, info = self.env.reset(seed=self.config['seed'])
        print(f"\n🔍 Initial observation shapes:")
        for agent, observation in obs.items():
            print(f"  {agent}: {observation.shape}")
    
    def setup_agents(self):
        """Setup the attacker and defender agents."""
        if self.env is None:
            raise ValueError("Environment must be setup before agents")
        
        # Get dimensions
        state_size = self.env.observation_spaces['attacker'].shape[0]
        attacker_action_size = self.env.action_spaces['attacker'].n
        defender_action_size = self.env.action_spaces['defender'].n
        
        print(f"\n🧠 Network Dimensions:")
        print(f"State size: {state_size}")
        print(f"Attacker actions: {attacker_action_size}")
        print(f"Defender actions: {defender_action_size}")
        
        # Create agents with epsilon exploration
        print("🤖 Initializing Actor-Critic Agents with Epsilon Exploration...")
        
        self.attacker_agent = ADTAgent(
            state_size=state_size,
            action_size=attacker_action_size,
            agent_name='attacker',
            lr=self.config['learning_rate'],
            gamma=self.config['gamma'],
            hidden_size=self.config['hidden_size'],
            device=self.config['device'],
            epsilon_start=self.config['epsilon_start'],
            epsilon_end=self.config['epsilon_end'],
            epsilon_decay=self.config['epsilon_decay']
        )
        
        self.defender_agent = ADTAgent(
            state_size=state_size,
            action_size=defender_action_size,
            agent_name='defender',
            lr=self.config['learning_rate'],
            gamma=self.config['gamma'],
            hidden_size=self.config['hidden_size'],
            device=self.config['device'],
            epsilon_start=self.config['epsilon_start'],
            epsilon_end=self.config['epsilon_end'],
            epsilon_decay=self.config['epsilon_decay']
        )
        
        print(f"✅ Agents initialized with epsilon exploration!")
        print(f"   Epsilon start: {self.config['epsilon_start']}")
        print(f"   Epsilon end: {self.config['epsilon_end']}")
        print(f"   Epsilon decay: {self.config['epsilon_decay']} steps")
    
    def get_available_actions(self, agent_name: str) -> list:
        """
        Get available actions for an agent.
        
        This method provides a fallback for environments that don't implement
        get_available_actions method.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of available actions
        """
        try:
            return self.env.get_available_actions(agent_name)
        except (AttributeError, NotImplementedError):
            # Fallback: return all possible actions
            if agent_name == 'attacker':
                return list(range(self.env.action_spaces['attacker'].n))
            else:
                return list(range(self.env.action_spaces['defender'].n))
    
    def check_episode_termination(self) -> Tuple[bool, str, bool]:
        """
        Check if episode should terminate and determine winner.
        
        This method can be overridden by subclasses for environment-specific logic.
        
        Returns:
            Tuple of (episode_terminated, winner, goal_achieved)
        """
        # Default implementation - check if goal is reached
        goal_achieved = getattr(self.env, 'goal_reached', False)
        winner = 'attacker' if goal_achieved else 'defender'
        return False, winner, goal_achieved  # Let environment handle termination
    
    def run_episode(self, episode_num: int, max_steps: int = 100) -> Dict[str, Any]:
        """
        Run a single episode and collect experiences.
        
        Args:
            episode_num: Episode number
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary containing episode data
        """
        obs, info = self.env.reset()
        
        episode_data = {
            'attacker_reward': 0,
            'defender_reward': 0,
            'length': 0,
            'winner': None,
            'goal_achieved': False
        }
        
        # Track experiences for each agent separately
        agent_experiences = {
            'attacker': [],
            'defender': []
        }
        
        step_count = 0
        
        for agent_name in self.env.agent_iter():
            if step_count >= max_steps:
                break
                
            observation, reward, termination, truncation, info = self.env.last()
            episode_data[f'{agent_name}_reward'] += reward
            
            if termination or truncation:
                # Store final experience with terminal flag
                if step_count > 0 and agent_experiences[agent_name]:
                    # Update last experience with final reward and terminal flag
                    last_exp = agent_experiences[agent_name][-1]
                    last_exp['reward'] = reward
                    last_exp['done'] = True
                break
            
            step_count += 1
            episode_data['length'] = step_count
            
            available_actions = self.get_available_actions(agent_name)
            
            # Get action from appropriate agent (with epsilon exploration)
            if agent_name == 'attacker':
                action, log_prob, value, _ = self.attacker_agent.get_action(
                    observation.astype(np.float32), available_actions, deterministic=False
                )
            else:
                action, log_prob, value, _ = self.defender_agent.get_action(
                    observation.astype(np.float32), available_actions, deterministic=False
                )
            
            # Skip if no valid action was returned
            if action is None or log_prob is None or value is None:
                self.env.step(0)  # Take a default action
                continue
            
            # Store experience
            agent_experiences[agent_name].append({
                'state': observation.copy(),
                'action': action,
                'reward': 0,  # Will be updated with next iteration's reward
                'log_prob': log_prob,
                'value': value,
                'done': False
            })
            
            self.env.step(action)
        
        # Store all experiences in agent buffers
        for agent_name, experiences in agent_experiences.items():
            agent = self.attacker_agent if agent_name == 'attacker' else self.defender_agent
            for exp in experiences:
                agent.store_transition(
                    state=exp['state'],
                    action=exp['action'],
                    reward=exp['reward'],
                    log_prob=exp['log_prob'],
                    value=exp['value'],
                    done=exp['done']
                )
        
        # Check final episode state
        _, winner, goal_achieved = self.check_episode_termination()
        episode_data['goal_achieved'] = goal_achieved
        episode_data['winner'] = winner
        
        return episode_data
    
    def print_training_stats(self, episode: int, start_time: float):
        """Print training statistics."""
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Get recent statistics
        recent_stats = self.metrics.get_recent_stats(window=100)
        
        print(f"\n{'='*80}")
        print(f"EPISODE {episode:4d} | Time: {elapsed_time/60:.1f}m")
        print(f"{'='*80}")
        
        if recent_stats:
            print(f"📊 Last 100 Episodes:")
            print(f"   🔥 Avg Attacker Reward: {recent_stats['avg_attacker_reward']:8.2f}")
            print(f"   🛡️ Avg Defender Reward: {recent_stats['avg_defender_reward']:8.2f}")
            print(f"   📏 Avg Episode Length:  {recent_stats['avg_episode_length']:8.2f} steps")
            print(f"   🏆 Attacker Win Rate:   {recent_stats['attacker_win_rate']:8.2%}")
            print(f"   🏆 Defender Win Rate:   {recent_stats['defender_win_rate']:8.2%}")
        
        # Show current epsilon values
        print(f"   🎯 Current Attacker ε:   {self.attacker_agent.get_epsilon():8.4f}")
        print(f"   🎯 Current Defender ε:   {self.defender_agent.get_epsilon():8.4f}")
        
        # Show recent losses if available
        if len(self.metrics.attacker_losses['total']) > 0:
            recent_att_loss = np.mean(self.metrics.attacker_losses['total'][-10:])
            recent_def_loss = np.mean(self.metrics.defender_losses['total'][-10:])
            print(f"   📉 Recent Attacker Loss: {recent_att_loss:8.4f}")
            print(f"   📉 Recent Defender Loss: {recent_def_loss:8.4f}")
    
    def train(self, env_class, env_spec_file: str) -> Tuple[Any, ADTAgent, ADTAgent, TrainingMetrics]:
        """
        Main training function.
        
        Args:
            env_class: The environment class to use for training
            env_spec_file: Path to the environment specification file
            
        Returns:
            Tuple of (environment, attacker_agent, defender_agent, metrics)
        """
        # Setup environment and agents
        self.setup_environment(env_class, env_spec_file)
        self.setup_agents()
        
        # Initialize training metrics
        self.metrics = TrainingMetrics()
        
        # Create directories for saving models and metrics
        if self.config.get('save_models', False):
            os.makedirs(self.config['model_dir'], exist_ok=True)
        if self.config.get('save_logs', False):
            os.makedirs(self.config['log_dir'], exist_ok=True)

        print(f"\n🚀 Starting Actor-Critic Training with Epsilon Exploration!")
        print(f"Max Episodes: {self.config['max_episodes']}")
        print(f"Update Frequency: {self.config['update_frequency']}")
        print(f"Print Frequency: {self.config['print_frequency']}")
        print("="*80)
        
        start_time = time.time()
        
        # Main training loop
        for episode in tqdm(range(1, self.config['max_episodes'] + 1), desc="Training", leave=False):
            # Run episode and collect experience
            episode_data = self.run_episode(episode, self.config['max_steps'])
            
            # Log episode data with epsilon values
            self.metrics.log_episode(
                episode_data, 
                attacker_epsilon=self.attacker_agent.get_epsilon(),
                defender_epsilon=self.defender_agent.get_epsilon()
            )
            
            # Update policies periodically
            if episode % self.config['update_frequency'] == 0:
                # Update attacker policy
                att_losses = self.attacker_agent.update_policy()
                if att_losses:
                    self.metrics.log_losses('attacker', att_losses)
                
                # Update defender policy
                def_losses = self.defender_agent.update_policy()
                if def_losses:
                    self.metrics.log_losses('defender', def_losses)
            
            # Print statistics
            if episode % self.config['print_frequency'] == 0:
                self.print_training_stats(episode, start_time)
        
        # Save final models and metrics
        print(f"\n💾 Saving final models and metrics...")
        if self.config.get('save_models', False):
            self.save_models("final")
        if self.config.get('save_logs', False):
            self.metrics.save_metrics(f"{self.config['log_dir']}/training_metrics.pkl")
        
        total_time = time.time() - start_time
        print(f"✅ Training completed in {total_time/60:.1f} minutes!")
        print(f"📊 Total episodes: {len(self.metrics.episode_lengths)}")
        
        # Print training summary instead of plotting
        print("📊 Training summary:")
        self.metrics.print_training_summary()
        
        return self.env, self.attacker_agent, self.defender_agent, self.metrics
    
    def save_models(self, suffix: str):
        """Save agent models with a given suffix."""
        if self.config.get('save_models', False):
            self.attacker_agent.save_model(f"{self.config['model_dir']}/attacker_{suffix}.pth")
            self.defender_agent.save_model(f"{self.config['model_dir']}/defender_{suffix}.pth")
    
    def load_models(self, suffix: str = "final"):
        """Load agent models with a given suffix."""
        if self.attacker_agent is None or self.defender_agent is None:
            raise ValueError("Agents must be setup before loading models")
        
        self.attacker_agent.load_model(f"{self.config['model_dir']}/attacker_{suffix}.pth")
        self.defender_agent.load_model(f"{self.config['model_dir']}/defender_{suffix}.pth")
    
    def test_models(self, n_test_games: int = 10) -> Dict[str, Any]:
        """
        Test the trained models with deterministic policies.
        
        Args:
            n_test_games: Number of test games to run
            
        Returns:
            Dictionary containing test results
        """
        if self.env is None or self.attacker_agent is None or self.defender_agent is None:
            raise ValueError("Environment and agents must be setup before testing")
        
        print(f"\n🧪 Testing Trained Models")
        print("="*50)
        print(f"Running {n_test_games} test games with trained models...")
        
        test_results = {
            'attacker_rewards': [],
            'defender_rewards': [],
            'episode_lengths': [],
            'winners': []
        }
        
        for game in range(n_test_games):
            obs, info = self.env.reset()
            
            game_rewards = {'attacker': 0, 'defender': 0}
            step_count = 0
            
            for agent_name in self.env.agent_iter():
                if step_count > 50:  # Max steps limit
                    break
                    
                observation, reward, termination, truncation, info = self.env.last()
                game_rewards[agent_name] += reward
                
                if termination or truncation:
                    self.env.step(None)
                    break
                
                step_count += 1
                
                # Get available actions
                available_actions = self.get_available_actions(agent_name)
                
                if not available_actions:
                    self.env.step(0)
                    continue
                
                # Get deterministic action from trained agent (no exploration)
                if agent_name == 'attacker':
                    action, _, _, _ = self.attacker_agent.get_action(observation, available_actions, deterministic=True)
                else:
                    action, _, _, _ = self.defender_agent.get_action(observation, available_actions, deterministic=True)
                
                self.env.step(action)
            
            # Store results
            test_results['attacker_rewards'].append(game_rewards['attacker'])
            test_results['defender_rewards'].append(game_rewards['defender'])
            test_results['episode_lengths'].append(step_count)
            
            # Determine winner
            _, winner, goal_achieved = self.check_episode_termination()
            test_results['winners'].append(winner)
                
            print(f"Game {game+1:2d}: Att {game_rewards['attacker']:6.1f} | "
                  f"Def {game_rewards['defender']:6.1f} | "
                  f"Length: {step_count:2d} | "
                  f"Winner: {test_results['winners'][-1]}")
        
        # Analyze test results
        self._print_test_summary(test_results)
        
        return test_results
    
    def _print_test_summary(self, test_results: Dict[str, Any]):
        """Print test results summary."""
        print(f"\n🎯 TEST RESULTS SUMMARY:")
        print(f"="*30)
        print(f"Average Attacker Reward: {np.mean(test_results['attacker_rewards']):.2f}")
        print(f"Average Defender Reward: {np.mean(test_results['defender_rewards']):.2f}")
        print(f"Average Episode Length: {np.mean(test_results['episode_lengths']):.2f} steps")

        attacker_wins = test_results['winners'].count('attacker')
        defender_wins = test_results['winners'].count('defender')  
        ties = test_results['winners'].count('tie')

        print(f"Attacker Wins: {attacker_wins}/{len(test_results['winners'])} ({attacker_wins/len(test_results['winners']):.1%})")
        print(f"Defender Wins: {defender_wins}/{len(test_results['winners'])} ({defender_wins/len(test_results['winners']):.1%})")
        print(f"Ties: {ties}/{len(test_results['winners'])} ({ties/len(test_results['winners']):.1%})")


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration dictionary."""
    return {
        # Environment settings
        'env_file': 'envs/adt_nuovo_env.json',
        
        # Training hyperparameters
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'hidden_size': 128,
        'max_episodes': 5000,
        'max_steps': 100,
        
        # Epsilon exploration parameters
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 5000,
        
        # Training control
        'update_frequency': 10,
        'print_frequency': 100,
        
        # System settings
        'device': 'cpu',
        'seed': 42,
        
        # Directory settings
        'model_dir': 'results/trained_models',
        'log_dir': 'results/training_logs',
        'save_models': False,
        'save_logs': False,
        
        # Testing
        'test_games': 10,
    }


def parse_training_args() -> Dict[str, Any]:
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description='Train ADT Environment with Actor-Critic and Epsilon Exploration')
    
    # Environment settings
    parser.add_argument('--env-file', type=str, default='envs/adt_nuovo_env.json',
                        help='Path to environment configuration file')
    
    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Size of hidden layers in neural networks')
    parser.add_argument('--max-episodes', type=int, default=5000,
                        help='Maximum number of training episodes')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Maximum steps per episode')
    
    # Epsilon exploration parameters
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Starting epsilon value for exploration')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                        help='Final epsilon value for exploration')
    parser.add_argument('--epsilon-decay', type=int, default=5000,
                        help='Number of steps for epsilon decay')
    
    # Training control
    parser.add_argument('--update-frequency', type=int, default=10,
                        help='Update policy every N episodes')
    parser.add_argument('--print-frequency', type=int, default=100,
                        help='Print statistics every N episodes')
    parser.add_argument('--save-frequency', type=int, default=1000,
                        help='Save intermediate models every N episodes')
    
    # System settings
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Directory settings
    parser.add_argument('--model-dir', type=str,
                        help='Directory to save trained models')
    parser.add_argument('--log-dir', type=str,
                        help='Directory to save training logs')
    
    # Testing
    parser.add_argument('--test-games', type=int, default=10,
                        help='Number of test games to run after training')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and only test existing models')
    parser.add_argument('--save-models', action='store_true',
                        help='Save models during training')
    parser.add_argument('--save-logs', action='store_true',
                        help='Save logs during training')

    args = parser.parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    return config
