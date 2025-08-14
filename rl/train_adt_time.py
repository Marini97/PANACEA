#!/usr/bin/env python3
"""
Standalone Training Script for Attack-Defense Tree Time Environment
with Actor-Critic agents and Epsilon-Greedy Exploration

This script provides a complete training pipeline for multi-agent reinforcement learning
in the ADT environment, including epsilon-greedy exploration to avoid local optima.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import argparse
from tqdm import tqdm

# Import our custom modules
from adt_time_env import AttackDefenseTreeMultiAgentTimeEnv
from adt_actor_critic import ADTAgent, TrainingMetrics

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def setup_environment_and_agents(config):
    """Setup the environment and initialize agents."""
    print("🌍 Setting up ADT Multi-Agent Environment...")
    
    # Create environment
    env = AttackDefenseTreeMultiAgentTimeEnv(config['env_file'], render_mode=None)
    print("✅ Environment created successfully!")
    
    # Environment information
    print(f"\n📊 Environment Information:")
    print(f"Agents: {env.possible_agents}")
    print(f"Action spaces: {env.action_spaces}")
    print(f"Observation spaces: {env.observation_spaces}")
    
    # Reset to get initial observations
    obs, info = env.reset(seed=config['seed'])
    print(f"\n🔍 Initial observation shapes:")
    for agent, observation in obs.items():
        print(f"  {agent}: {observation.shape}")
    
    # Get dimensions
    state_size = env.observation_spaces['attacker'].shape[0]
    attacker_action_size = env.action_spaces['attacker'].n
    defender_action_size = env.action_spaces['defender'].n
    
    print(f"\n🧠 Network Dimensions:")
    print(f"State size: {state_size}")
    print(f"Attacker actions: {attacker_action_size}")
    print(f"Defender actions: {defender_action_size}")
    
    # Create agents with epsilon exploration
    print("🤖 Initializing Actor-Critic Agents with Epsilon Exploration...")
    
    attacker_agent = ADTAgent(
        state_size=state_size,
        action_size=attacker_action_size,
        agent_name='attacker',
        lr=config['learning_rate'],
        gamma=config['gamma'],
        hidden_size=config['hidden_size'],
        device=config['device'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay']
    )
    
    defender_agent = ADTAgent(
        state_size=state_size,
        action_size=defender_action_size,
        agent_name='defender',
        lr=config['learning_rate'],
        gamma=config['gamma'],
        hidden_size=config['hidden_size'],
        device=config['device'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay']
    )
    
    print(f"✅ Agents initialized with epsilon exploration!")
    print(f"   Epsilon start: {config['epsilon_start']}")
    print(f"   Epsilon end: {config['epsilon_end']}")
    print(f"   Epsilon decay: {config['epsilon_decay']} steps")
    
    return env, attacker_agent, defender_agent


def run_episode(env, attacker_agent, defender_agent, episode_num, max_steps=100):
    """Run a single episode and collect experiences."""
    obs, info = env.reset()
    
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
    
    for agent_name in env.agent_iter():
        if step_count >= max_steps:
            break
            
        observation, reward, termination, truncation, info = env.last()
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
        
        available_actions = env.get_available_actions(agent_name)
        
        # Get action from appropriate agent (with epsilon exploration)
        if agent_name == 'attacker':
            action, log_prob, value, _ = attacker_agent.get_action(
                observation.astype(np.float32), available_actions, deterministic=False
            )
        else:
            action, log_prob, value, _ = defender_agent.get_action(
                observation.astype(np.float32), available_actions, deterministic=False
            )
        
        # Store experience
        agent_experiences[agent_name].append({
            'state': observation.copy(),
            'action': action,
            'reward': 0,  # Will be updated with next iteration's reward
            'log_prob': log_prob,
            'value': value,
            'done': False
        })
        
        env.step(action)
    
    # Store all experiences in agent buffers
    for agent_name, experiences in agent_experiences.items():
        agent = attacker_agent if agent_name == 'attacker' else defender_agent
        for exp in experiences:
            agent.store_transition(
                state=exp['state'],
                action=exp['action'],
                reward=exp['reward'],
                log_prob=exp['log_prob'],
                value=exp['value'],
                done=exp['done']
            )
    
    episode_data['goal_achieved'] = env.goal_reached
    episode_data['winner'] = 'attacker' if episode_data['goal_achieved'] else 'defender'
    
    return episode_data


def print_training_stats(episode, metrics, attacker_agent, defender_agent, start_time):
    """Print training statistics."""
    current_time = time.time()
    elapsed_time = current_time - start_time
    
    # Get recent statistics
    recent_stats = metrics.get_recent_stats(window=100)
    
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
    print(f"   🎯 Current Attacker ε:   {attacker_agent.get_epsilon():8.4f}")
    print(f"   🎯 Current Defender ε:   {defender_agent.get_epsilon():8.4f}")
    
    # Show recent losses if available
    if len(metrics.attacker_losses['total']) > 0:
        recent_att_loss = np.mean(metrics.attacker_losses['total'][-10:])
        recent_def_loss = np.mean(metrics.defender_losses['total'][-10:])
        print(f"   📉 Recent Attacker Loss: {recent_att_loss:8.4f}")
        print(f"   📉 Recent Defender Loss: {recent_def_loss:8.4f}")


def train_agents(config):
    """Main training function."""
    # Set random seeds for reproducibility
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Setup environment and agents
    env, attacker_agent, defender_agent = setup_environment_and_agents(config)
    
    # Initialize training metrics
    metrics = TrainingMetrics()
    
    # Create directories for saving models and metrics
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    print(f"\n🚀 Starting Actor-Critic Training with Epsilon Exploration!")
    print(f"Max Episodes: {config['max_episodes']}")
    print(f"Update Frequency: {config['update_frequency']}")
    print(f"Print Frequency: {config['print_frequency']}")
    print("="*80)
    
    start_time = time.time()
    
    # Main training loop
    for episode in tqdm(range(1, config['max_episodes'] + 1), desc="Training"):
        # Run episode and collect experience
        episode_data = run_episode(env, attacker_agent, defender_agent, episode, config['max_steps'])
        
        # Log episode data with epsilon values
        metrics.log_episode(
            episode_data, 
            attacker_epsilon=attacker_agent.get_epsilon(),
            defender_epsilon=defender_agent.get_epsilon()
        )
        
        # Update policies periodically
        if episode % config['update_frequency'] == 0:
            # Update attacker policy
            att_losses = attacker_agent.update_policy()
            if att_losses:
                metrics.log_losses('attacker', att_losses)
            
            # Update defender policy
            def_losses = defender_agent.update_policy()
            if def_losses:
                metrics.log_losses('defender', def_losses)
        
        # Print statistics
        if episode % config['print_frequency'] == 0:
            print_training_stats(episode, metrics, attacker_agent, defender_agent, start_time)
        
        # Save intermediate models
        if episode % config['save_frequency'] == 0:
            attacker_agent.save_model(f"{config['model_dir']}/attacker_episode_{episode}.pth")
            defender_agent.save_model(f"{config['model_dir']}/defender_episode_{episode}.pth")
    
    # Save final models and metrics
    print(f"\n💾 Saving final models and metrics...")
    attacker_agent.save_model(f"{config['model_dir']}/time_attacker_actor_critic.pth")
    defender_agent.save_model(f"{config['model_dir']}/time_defender_actor_critic.pth")
    metrics.save_metrics(f"{config['log_dir']}/time_training_metrics.pkl")
    
    total_time = time.time() - start_time
    print(f"✅ Training completed in {total_time/60:.1f} minutes!")
    print(f"📊 Total episodes: {len(metrics.episode_lengths)}")
    
    # Generate training visualization
    print("📊 Generating training visualization...")
    metrics.plot_training_progress(save_path=f"{config['log_dir']}/time_training_progress.png")
    
    return env, attacker_agent, defender_agent, metrics


def test_trained_models(env, attacker_agent, defender_agent, n_test_games=10):
    """Test the trained models with deterministic policies."""
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
        obs, info = env.reset()
        
        game_rewards = {'attacker': 0, 'defender': 0}
        step_count = 0
        
        for agent_name in env.agent_iter():
            if step_count > 50:  # Max steps limit
                break
                
            observation, reward, termination, truncation, info = env.last()
            game_rewards[agent_name] += reward
            
            if termination or truncation:
                env.step(None)
                break
            
            step_count += 1
            
            # Get available actions
            try:
                available_actions = env.get_available_actions(agent_name)
            except:
                # Fallback if method doesn't exist
                if agent_name == 'attacker':
                    available_actions = list(range(6))
                else:
                    available_actions = list(range(7))
            
            if not available_actions:
                env.step(0)
                continue
            
            # Get deterministic action from trained agent (no exploration)
            if agent_name == 'attacker':
                action, _, _, _ = attacker_agent.get_action(observation, available_actions, deterministic=True)
            else:
                action, _, _, _ = defender_agent.get_action(observation, available_actions, deterministic=True)
            
            env.step(action)
        
        # Store results
        test_results['attacker_rewards'].append(game_rewards['attacker'])
        test_results['defender_rewards'].append(game_rewards['defender'])
        test_results['episode_lengths'].append(step_count)
        
        # Determine winner
        if env.goal_reached:
            test_results['winners'].append('attacker')
        else:
            test_results['winners'].append('defender')
            
        print(f"Game {game+1:2d}: Att {game_rewards['attacker']:6.1f} | "
              f"Def {game_rewards['defender']:6.1f} | "
              f"Length: {step_count:2d} | "
              f"Winner: {test_results['winners'][-1]}")
    
    # Analyze test results
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

    return test_results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train ADT Time Environment with Actor-Critic and Epsilon Exploration')
    
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
    parser.add_argument('--model-dir', type=str, default='trained_models',
                        help='Directory to save trained models')
    parser.add_argument('--log-dir', type=str, default='training_logs',
                        help='Directory to save training logs')
    
    # Testing
    parser.add_argument('--test-games', type=int, default=10,
                        help='Number of test games to run after training')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and only test existing models')
    
    args = parser.parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    print("🚀 ADT Time Environment Training with Epsilon Exploration")
    print("="*60)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    print("="*60)
    
    if not config['skip_training']:
        # Train the agents
        env, attacker_agent, defender_agent, metrics = train_agents(config)
        
        # Test the trained models
        test_results = test_trained_models(env, attacker_agent, defender_agent, config['test_games'])
    else:
        print("⏭️ Skipping training, loading existing models for testing...")
        # Load existing models and test
        env, attacker_agent, defender_agent = setup_environment_and_agents(config)
        
        try:
            attacker_agent.load_model(f"{config['model_dir']}/time_attacker_actor_critic.pth")
            defender_agent.load_model(f"{config['model_dir']}/time_defender_actor_critic.pth")
            test_results = test_trained_models(env, attacker_agent, defender_agent, config['test_games'])
        except FileNotFoundError:
            print("❌ No trained models found. Please run training first.")
            return
    
    print(f"\n✅ Training and testing completed successfully!")


if __name__ == "__main__":
    main()
