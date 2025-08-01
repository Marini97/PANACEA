"""
Test Trained ADT Models - Step by Step Game Analysis

This script loads the trained actor-critic models and allows you to observe
games step by step, analyzing the decision-making process of both agents.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import json
import os
from adt_env import AttackDefenseTreeEnv


class ActorCritic(nn.Module):
    """Actor-Critic network for the ADT environment."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(ActorCritic, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared layers with proper initialization
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head (policy network) - outputs logits, not probabilities
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Initialize actor head with smaller weights for stability
        nn.init.orthogonal_(self.actor[-1].weight, gain=1)
        
    def forward(self, state):
        """Forward pass through the network."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        shared_out = self.shared_layers(state)
        
        # Get policy logits (not probabilities)
        logits = self.actor(shared_out)
        
        # Get state value
        value = self.critic(shared_out)
        
        return logits, value
    
    def get_action(self, state, available_actions=None):
        """Sample an action from the policy."""
        logits, value = self.forward(state)
        
        # Apply mask to logits before softmax
        if available_actions is not None and len(available_actions) > 0:
            # Create a large negative mask for unavailable actions
            mask = torch.full_like(logits, -1e9)
            
            # Convert available_actions to tensor if it's a list
            if isinstance(available_actions, list):
                available_actions = torch.tensor(available_actions, dtype=torch.long)
            
            # Handle batch dimension properly - get batch size from logits
            batch_size = logits.shape[0]
            
            # Set mask to 0 for available actions for all batches
            for batch_idx in range(batch_size):
                # Ensure available_actions indices are within bounds
                valid_actions = available_actions[available_actions < logits.shape[1]]
                if len(valid_actions) > 0:
                    mask[batch_idx, valid_actions] = 0
            
            logits = logits + mask
        
        # Clamp logits to prevent extreme values
        logits = torch.clamp(logits, min=-20, max=20)
        
        # Convert to probabilities using softmax
        policy = F.softmax(logits, dim=-1)
        
        # Add small epsilon to prevent zero probabilities
        policy = policy + 1e-8
        policy = policy / policy.sum(dim=-1, keepdim=True)
        
        # Sample action
        dist = Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze(), policy.squeeze()


class TrainedAgent:
    """Wrapper for trained agent models."""
    
    def __init__(self, model_path: str, state_size: int, action_size: int):
        self.network = ActorCritic(state_size, action_size)
        
        # Load trained model
        checkpoint = torch.load(model_path, map_location='cpu')
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.eval()  # Set to evaluation mode
        
        self.state_size = state_size
        self.action_size = action_size
        
        print(f"Loaded trained model from {model_path}")
        print(f"State size: {state_size}, Action size: {action_size}")
    
    def get_action(self, state, available_actions=None, deterministic=False):
        """Get action from the trained policy."""
        if isinstance(state, list):
            state = np.array(state)
        state_tensor = torch.FloatTensor(state)
        
        with torch.no_grad():
            action, log_prob, value, policy = self.network.get_action(state_tensor, available_actions)
            
            if deterministic and available_actions:
                # Choose the action with highest probability among available actions
                masked_policy = policy.clone()
                for i in range(len(policy)):
                    if i not in available_actions:
                        masked_policy[i] = 0
                action = torch.argmax(masked_policy).item()
        
        return action, log_prob, value, policy
    
    def get_action_probabilities(self, state, available_actions=None):
        """Get action probabilities for analysis."""
        if isinstance(state, list):
            state = np.array(state)
        state_tensor = torch.FloatTensor(state)
        
        with torch.no_grad():
            logits, value = self.network(state_tensor)
            
            # Apply mask for available actions
            if available_actions is not None and len(available_actions) > 0:
                mask = torch.full_like(logits, -1e9)
                for action in available_actions:
                    if action < logits.shape[1]:
                        mask[0, action] = 0
                logits = logits + mask
            
            policy = F.softmax(logits, dim=-1).squeeze()
            
        return policy.numpy(), value.item()


def load_trained_models(models_dir="/home/valerio/PANACEA/trained_models"):
    """Load both trained models."""
    # Load environment to get dimensions
    env = AttackDefenseTreeEnv("/home/valerio/PANACEA/gym/envs/adt_nuovo_env.json")
    
    # Check if observation_space exists and has shape attribute
    if hasattr(env.observation_space, 'shape') and env.observation_space.shape:
        state_size = env.observation_space.shape[0]
    else:
        # Fallback: get state size from state variables
        state_size = len(env.state_vars)
    
    # Load attacker model
    attacker_path = os.path.join(models_dir, "attacker_actor_critic.pth")
    attacker_agent = TrainedAgent(attacker_path, state_size, env.num_attacker_actions)
    
    # Load defender model
    defender_path = os.path.join(models_dir, "defender_actor_critic.pth")
    defender_agent = TrainedAgent(defender_path, state_size, env.num_defender_actions)
    
    return attacker_agent, defender_agent, env


def analyze_game_step(env, attacker_agent, defender_agent, state, current_player, step_num):
    """Analyze a single game step in detail."""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {current_player.upper()}'S TURN")
    print(f"{'='*60}")
    
    # Show current state
    print("\n🎯 CURRENT GAME STATE:")
    print(f"Goal ({env.goal}): {env.state[env.goal]}")
    
    # Show key state variables
    key_vars = [var for var in env.state_vars if var not in ["current_player"]]
    for i, var in enumerate(key_vars):
        value = state[i] if i < len(state) else env.state.get(var, 0)
        status = "🔴" if value == 0 else "🟡" if value == 1 else "🟢"
        print(f"  {status} {var}: {value}")
    
    # Get available actions
    available_actions = env.get_available_actions()
    print(f"\n🎮 AVAILABLE ACTIONS: {available_actions}")
    
    # Get agent and action space
    if current_player == 'attacker':
        agent = attacker_agent
        actions_dict = env.attacker_actions
        print("🔥 ATTACKER ANALYSIS:")
    else:
        agent = defender_agent
        actions_dict = env.defender_actions
        print("🛡️  DEFENDER ANALYSIS:")
    
    # Get action probabilities and state value
    policy, state_value = agent.get_action_probabilities(state, available_actions)
    print(f"State Value Estimate: {state_value:.3f}")
    
    # Show available actions with probabilities
    print("\n📊 ACTION PROBABILITIES:")
    if available_actions:
        action_probs = []
        for action_id in available_actions:
            if str(action_id) in actions_dict:
                action_name = actions_dict[str(action_id)]['name']
                action_cost = actions_dict[str(action_id)].get('cost', 0)
                prob = policy[action_id] if action_id < len(policy) else 0.0
                action_probs.append((action_id, action_name, action_cost, prob))
        
        # Sort by probability (descending)
        action_probs.sort(key=lambda x: x[3], reverse=True)
        
        for rank, (action_id, action_name, cost, prob) in enumerate(action_probs, 1):
            prob_bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            print(f"  {rank}. [{action_id:2d}] {action_name:20s} (cost:{cost:3d}) {prob:.3f} |{prob_bar}|")
    else:
        print("  ❌ No valid actions available!")
    
    # Get the chosen action
    chosen_action, _, _, _ = agent.get_action(state, available_actions, deterministic=False)
    
    return chosen_action, available_actions, state_value, policy


def step_by_step_game(attacker_agent, defender_agent, env, max_steps=20, auto_mode=False):
    """Play a game step by step with detailed analysis."""
    print("🎮 STARTING STEP-BY-STEP GAME ANALYSIS")
    print("=" * 80)
    
    obs, info = env.reset()
    step_num = 1
    total_rewards = {'attacker': 0, 'defender': 0}
    game_history = []
    
    while step_num <= max_steps:
        current_player = info['current_player']
        
        # Analyze current step
        chosen_action, available_actions, state_value, policy = analyze_game_step(
            env, attacker_agent, defender_agent, obs, current_player, step_num
        )
        
        # Get action name
        if current_player == 'attacker':
            action_name = env.attacker_actions.get(str(chosen_action), {}).get('name', f'unknown_action_{chosen_action}')
        else:
            action_name = env.defender_actions.get(str(chosen_action), {}).get('name', f'unknown_action_{chosen_action}')
        
        print(f"\n🎯 CHOSEN ACTION: {chosen_action} ({action_name})")
        
        # Record step
        game_history.append({
            'step': step_num,
            'player': current_player,
            'action_id': chosen_action,
            'action_name': action_name,
            'state_value': state_value,
            'available_actions': available_actions.copy(),
            'state': obs.copy()
        })
        
        # Execute action
        next_obs, reward, terminated, truncated, next_info = env.step(chosen_action)
        
        total_rewards[current_player] += reward
        print(f"💰 REWARD: {reward} (Total {current_player}: {total_rewards[current_player]})")
        
        if terminated or truncated:
            print(f"\n🏁 GAME ENDED!")
            winner = next_info.get('winner', 'unknown')
            termination_reason = next_info.get('termination_reason', 'unknown')
            
            if winner:
                print(f"🏆 WINNER: {winner.upper()}")
            else:
                print("🏆 WINNER: UNKNOWN")
            print(f"📋 REASON: {termination_reason}")
            print(f"🎯 GOAL STATE: {env.state[env.goal]}")
            
            # Handle terminal rewards
            if winner == 'attacker':
                defender_penalty = next_info.get('defender_reward', 0)
                total_rewards['defender'] += defender_penalty
                print(f"💔 Defender terminal penalty: {defender_penalty}")
            
            print(f"\n📊 FINAL SCORES:")
            print(f"  🔥 Attacker: {total_rewards['attacker']}")
            print(f"  🛡️  Defender: {total_rewards['defender']}")
            
            break
        
        # Update for next iteration
        obs = next_obs
        info = next_info
        step_num += 1
        
        # Wait for user input if not in auto mode
        if not auto_mode:
            user_input = input("\n⏯️  Press Enter to continue, 'a' for auto mode, or 'q' to quit: ").strip().lower()
            if user_input == 'q':
                print("Game terminated by user.")
                break
            elif user_input == 'a':
                auto_mode = True
                print("Switched to auto mode...")
        else:
            import time
            time.sleep(1)  # Small delay in auto mode
    
    return game_history, total_rewards


def compare_agents_on_state(attacker_agent, defender_agent, env, num_samples=3):
    """Compare how agents evaluate different game states."""
    print("\n🔍 AGENT STATE EVALUATION COMPARISON")
    print("=" * 60)
    
    for sample in range(num_samples):
        obs, info = env.reset()
        
        # Get both agents' evaluations of the same state
        print(f"\n📊 Sample State {sample + 1}:")
        
        # Show state
        print("State variables:")
        for i, var in enumerate(env.state_vars):
            if i < len(obs):
                print(f"  {var}: {obs[i]}")
        
        # Attacker's perspective
        print("\n🔥 ATTACKER'S PERSPECTIVE:")
        att_available = env.get_available_actions() if info['current_player'] == 'attacker' else []
        if att_available:
            att_policy, att_value = attacker_agent.get_action_probabilities(obs, att_available)
            print(f"State Value: {att_value:.3f}")
            top_actions = np.argsort(att_policy)[-3:][::-1]
            for i, action_id in enumerate(top_actions):
                if action_id in att_available and str(action_id) in env.attacker_actions:
                    action_name = env.attacker_actions[str(action_id)]['name']
                    print(f"  {i+1}. {action_name}: {att_policy[action_id]:.3f}")
        
        # Switch to defender's turn to evaluate
        if info['current_player'] == 'attacker':
            env.state['current_player'] = 1
        
        def_available = env.get_available_actions()
        print("\n🛡️  DEFENDER'S PERSPECTIVE:")
        if def_available:
            def_policy, def_value = defender_agent.get_action_probabilities(obs, def_available)
            print(f"State Value: {def_value:.3f}")
            top_actions = np.argsort(def_policy)[-3:][::-1]
            for i, action_id in enumerate(top_actions):
                if action_id in def_available and str(action_id) in env.defender_actions:
                    action_name = env.defender_actions[str(action_id)]['name']
                    print(f"  {i+1}. {action_name}: {def_policy[action_id]:.3f}")


def main():
    """Main function to run the step-by-step game analysis."""
    print("🤖 LOADING TRAINED ADT MODELS...")
    
    try:
        attacker_agent, defender_agent, env = load_trained_models()
        print("✅ Models loaded successfully!")
        
        while True:
            print("\n" + "=" * 80)
            print("🎮 ADT TRAINED MODELS ANALYZER")
            print("=" * 80)
            print("1. Play step-by-step game")
            print("2. Compare agents on random states")
            print("3. Quick auto game")
            print("4. Exit")
            
            choice = input("\nChoose an option (1-4): ").strip()
            
            if choice == '1':
                print("\n🎯 Starting interactive step-by-step game...")
                game_history, final_rewards = step_by_step_game(
                    attacker_agent, defender_agent, env, auto_mode=False
                )
                
            elif choice == '2':
                compare_agents_on_state(attacker_agent, defender_agent, env)
                
            elif choice == '3':
                print("\n⚡ Running quick auto game...")
                game_history, final_rewards = step_by_step_game(
                    attacker_agent, defender_agent, env, auto_mode=True
                )
                
            elif choice == '4':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please try again.")
                
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Make sure you have trained models in the trained_models directory.")


if __name__ == "__main__":
    main()
