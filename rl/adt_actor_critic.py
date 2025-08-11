"""
Actor-Critic Network Implementation for Attack-Defense Tree Multi-Agent Environment

This module contains the neural network architecture and training utilities
for the multi-agent reinforcement learning in the ADT environment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import pickle


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
                available_actions = torch.tensor(available_actions, dtype=torch.long, device=logits.device)
            
            # Handle batch dimension properly - get batch size from logits
            batch_size = logits.shape[0]
            
            # Set mask to 0 for available actions for all batches
            for batch_idx in range(batch_size):
                # Ensure available_actions indices are within bounds
                valid_actions = available_actions[available_actions < logits.shape[1]]
                if len(valid_actions) > 0:
                    mask[batch_idx, valid_actions] = 0
            
            logits = logits + mask
        else:
            return None, None, None, None
        
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


class ADTAgent:
    """Actor-Critic agent for the ADT multi-agent environment."""
    
    def __init__(self, state_size: int, action_size: int, agent_name: str, 
                 lr: float = 3e-4, gamma: float = 0.99, hidden_size: int = 128, device: str = "cpu"):
        self.agent_name = agent_name
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.device = torch.device(device)
        
        # Neural network
        self.network = ActorCritic(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Training data storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.total_losses = []
        
    def get_action(self, state, available_actions=None, deterministic=False):
        """Get action from the policy."""
        if isinstance(state, list):
            state = np.array(state)
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # For training, we need gradients
        action, log_prob, value, policy = self.network.get_action(state_tensor, available_actions)
        
        if policy != None and deterministic and available_actions:
            # Choose the action with highest probability among available actions
            with torch.no_grad():
                masked_policy = policy.clone()
                for i in range(len(policy)):
                    if i not in available_actions:
                        masked_policy[i] = 0
                action = torch.argmax(masked_policy).item()
            # Recalculate log_prob for the chosen action (ensure action is int)
            action_idx = int(action)
            log_prob = torch.log(policy[action_idx] + 1e-8)
        
        return action, log_prob, value, policy
    
    def get_action_details(self, state, available_actions=None):
        """Get detailed action information for analysis."""
        if isinstance(state, list):
            state = np.array(state)
        state_tensor = torch.FloatTensor(state).to(self.device)
        
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
            
            # Get the most likely action
            action = torch.argmax(policy).item()
            
        return action, policy.cpu().numpy(), value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store a transition for training."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        
        # Ensure consistent tensor shapes
        if isinstance(log_prob, torch.Tensor):
            if log_prob.dim() == 0:  # Scalar tensor
                log_prob = log_prob.unsqueeze(0)  # Make it [1]
            self.log_probs.append(log_prob.cpu())
        else:
            self.log_probs.append(torch.tensor([log_prob], dtype=torch.float32))
            
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:  # Scalar tensor
                value = value.unsqueeze(0)  # Make it [1]
            self.values.append(value.cpu())
        else:
            self.values.append(torch.tensor([value], dtype=torch.float32))
            
        self.dones.append(done)
    
    def compute_discounted_rewards(self):
        """Compute discounted rewards (returns)."""
        returns = []
        discounted_sum = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
            
        return returns
    
    def update_policy(self):
        """Update the policy using actor-critic algorithm."""
        if len(self.states) == 0:
            return
            
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        
        # Handle log_probs and values with proper tensor stacking
        try:
            log_probs = torch.stack(self.log_probs).squeeze().to(self.device)
            values = torch.stack(self.values).squeeze().to(self.device)
        except RuntimeError as e:
            print(f"Tensor stacking error: {e}")
            print(f"log_probs shapes: {[lp.shape if hasattr(lp, 'shape') else type(lp) for lp in self.log_probs]}")
            print(f"values shapes: {[v.shape if hasattr(v, 'shape') else type(v) for v in self.values]}")
            # Clear memory and return None
            self.clear_episode_data()
            return None
        
        # Compute returns
        returns = self.compute_discounted_rewards()
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute advantages
        advantages = returns - values
        
        # Actor loss (policy gradient)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(values, returns)
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Store losses for monitoring
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.total_losses.append(total_loss.item())
        
        # Clear episode data
        self.clear_episode_data()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def clear_episode_data(self):
        """Clear stored episode data."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def clear_memory(self):
        """Alias for clear_episode_data for consistency."""
        self.clear_episode_data()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'agent_name': self.agent_name,
            'state_size': self.state_size,
            'action_size': self.action_size
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.network.eval()
        print(f"Model loaded from {filepath}")


class TrainingMetrics:
    """Class to track and visualize training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.attacker_rewards = []
        self.defender_rewards = []
        self.episode_lengths = []
        self.attacker_wins = []
        self.defender_wins = []
        self.attacker_losses = {'actor': [], 'critic': [], 'total': []}
        self.defender_losses = {'actor': [], 'critic': [], 'total': []}
        
    def log_episode(self, episode_data: Dict):
        """Log data from a completed episode."""
        self.attacker_rewards.append(episode_data.get('attacker_reward', 0))
        self.defender_rewards.append(episode_data.get('defender_reward', 0))
        self.episode_lengths.append(episode_data.get('length', 0))
        
        # Determine winner
        att_reward = episode_data.get('attacker_reward', 0)
        def_reward = episode_data.get('defender_reward', 0)
        
        if att_reward > def_reward:
            self.attacker_wins.append(1)
            self.defender_wins.append(0)
        else:
            self.attacker_wins.append(0)
            self.defender_wins.append(1)
    
    def log_losses(self, agent_name: str, losses: Dict):
        """Log training losses."""
        if agent_name == 'attacker':
            self.attacker_losses['actor'].append(losses['actor_loss'])
            self.attacker_losses['critic'].append(losses['critic_loss'])
            self.attacker_losses['total'].append(losses['total_loss'])
        else:
            self.defender_losses['actor'].append(losses['actor_loss'])
            self.defender_losses['critic'].append(losses['critic_loss'])
            self.defender_losses['total'].append(losses['total_loss'])
    
    def get_recent_stats(self, window: int = 100):
        """Get statistics for the most recent episodes."""
        if len(self.attacker_rewards) < window:
            window = len(self.attacker_rewards)
        
        if window == 0:
            return {}
        
        recent_att_rewards = self.attacker_rewards[-window:]
        recent_def_rewards = self.defender_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        recent_att_wins = self.attacker_wins[-window:]
        recent_def_wins = self.defender_wins[-window:]
        
        return {
            'avg_attacker_reward': np.mean(recent_att_rewards),
            'avg_defender_reward': np.mean(recent_def_rewards),
            'avg_episode_length': np.mean(recent_lengths),
            'attacker_win_rate': np.mean(recent_att_wins),
            'defender_win_rate': np.mean(recent_def_wins),
            'episodes': window
        }
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot comprehensive training progress."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Actor-Critic Training Progress', fontsize=16)
        
        # Moving average function
        def moving_average(data, window=100):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Episode rewards
        axes[0, 0].plot(moving_average(self.attacker_rewards), label='Attacker', color='red', alpha=0.8)
        axes[0, 0].plot(moving_average(self.defender_rewards), label='Defender', color='blue', alpha=0.8)
        axes[0, 0].set_title('Episode Rewards (Moving Average)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Win rates
        win_window = 50
        att_win_rate = moving_average([np.mean(self.attacker_wins[max(0, i-win_window):i+1]) 
                                      for i in range(len(self.attacker_wins))])
        def_win_rate = moving_average([np.mean(self.defender_wins[max(0, i-win_window):i+1]) 
                                      for i in range(len(self.defender_wins))])
        
        axes[0, 1].plot(att_win_rate, label='Attacker', color='red', alpha=0.8)
        axes[0, 1].plot(def_win_rate, label='Defender', color='blue', alpha=0.8)
        axes[0, 1].set_title(f'Win Rates (Rolling {win_window}-episode window)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[0, 2].plot(moving_average(self.episode_lengths), color='green', alpha=0.8)
        axes[0, 2].set_title('Episode Lengths (Moving Average)')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Steps')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Training losses - Attacker
        if self.attacker_losses['total']:
            axes[1, 0].plot(moving_average(self.attacker_losses['actor']), 
                           label='Actor', color='orange', alpha=0.8)
            axes[1, 0].plot(moving_average(self.attacker_losses['critic']), 
                           label='Critic', color='purple', alpha=0.8)
            axes[1, 0].plot(moving_average(self.attacker_losses['total']), 
                           label='Total', color='red', alpha=0.8)
        axes[1, 0].set_title('Attacker Training Losses')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training losses - Defender
        if self.defender_losses['total']:
            axes[1, 1].plot(moving_average(self.defender_losses['actor']), 
                           label='Actor', color='orange', alpha=0.8)
            axes[1, 1].plot(moving_average(self.defender_losses['critic']), 
                           label='Critic', color='purple', alpha=0.8)
            axes[1, 1].plot(moving_average(self.defender_losses['total']), 
                           label='Total', color='blue', alpha=0.8)
        axes[1, 1].set_title('Defender Training Losses')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Final performance distribution
        if len(self.attacker_rewards) > 0:
            recent_window = min(200, len(self.attacker_rewards))
            recent_att = self.attacker_rewards[-recent_window:]
            recent_def = self.defender_rewards[-recent_window:]

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
            print(f"Training progress plot saved to {save_path}")
        
        plt.show()
    
    def save_metrics(self, filepath: str):
        """Save training metrics to file."""
        metrics_data = {
            'attacker_rewards': self.attacker_rewards,
            'defender_rewards': self.defender_rewards,
            'episode_lengths': self.episode_lengths,
            'attacker_wins': self.attacker_wins,
            'defender_wins': self.defender_wins,
            'attacker_losses': self.attacker_losses,
            'defender_losses': self.defender_losses
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(metrics_data, f)
        
        print(f"Training metrics saved to {filepath}")
    
    def load_metrics(self, filepath: str):
        """Load training metrics from file."""
        with open(filepath, 'rb') as f:
            metrics_data = pickle.load(f)
        
        self.attacker_rewards = metrics_data['attacker_rewards']
        self.defender_rewards = metrics_data['defender_rewards']
        self.episode_lengths = metrics_data['episode_lengths']
        self.attacker_wins = metrics_data['attacker_wins']
        self.defender_wins = metrics_data['defender_wins']
        self.attacker_losses = metrics_data['attacker_losses']
        self.defender_losses = metrics_data['defender_losses']
        
        print(f"Training metrics loaded from {filepath}")
