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
                 lr: float = 3e-4, gamma: float = 0.99, hidden_size: int = 128, device: str = "cpu",
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: int = 5000):
        self.agent_name = agent_name
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.device = torch.device(device)
        
        # Epsilon-greedy exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.steps_done = 0
        
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
        """Get action from the policy with epsilon-greedy exploration."""
        if isinstance(state, list):
            state = np.array(state)
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Update epsilon for exploration
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-1. * self.steps_done / self.epsilon_decay)
        
        # Epsilon-greedy exploration (only during training, not when deterministic)
        if not deterministic and np.random.random() < self.epsilon and available_actions:
            # Random action from available actions
            action = np.random.choice(available_actions)
            
            # Still need to get policy values for consistency - KEEP GRADIENTS for training
            logits, value = self.network(state_tensor)
            
            # Apply mask for available actions
            if available_actions is not None and len(available_actions) > 0:
                mask = torch.full_like(logits, -1e9)
                for avail_action in available_actions:
                    if avail_action < logits.shape[1]:
                        mask[0, avail_action] = 0
                logits = logits + mask
            
            policy = F.softmax(logits, dim=-1).squeeze()
            log_prob = torch.log(policy[action] + 1e-8)
            
            return action, log_prob, value.squeeze(), policy
        
        # Normal policy-based action selection
        action, log_prob, value, policy = self.network.get_action(state_tensor, available_actions)
        
        if policy is not None and deterministic and available_actions:
            # Choose the action with highest probability among available actions
            with torch.no_grad():
                masked_policy = policy.clone()
                for i in range(len(policy)):
                    if i not in available_actions:
                        masked_policy[i] = 0
                action = torch.argmax(masked_policy).item()
            # Recalculate log_prob for the chosen action
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
    
    def get_epsilon(self):
        """Get current epsilon value for monitoring."""
        return self.epsilon
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store a transition for training."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        
        # Keep tensors on-device and with gradients for backprop during update
        if isinstance(log_prob, torch.Tensor):
            if log_prob.dim() == 0:
                log_prob = log_prob.unsqueeze(0)
            self.log_probs.append(log_prob)  # do NOT move to CPU or detach
        else:
            self.log_probs.append(torch.tensor([log_prob], dtype=torch.float32, device=self.device))
        
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                value = value.unsqueeze(0)
            self.values.append(value)  # do NOT move to CPU or detach
        else:
            self.values.append(torch.tensor([value], dtype=torch.float32, device=self.device))
        
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
        if len(self.states) < 2:
            return None

        # Stack log_probs and values (keep gradients)
        try:
            log_probs = torch.stack([
                lp.squeeze() if isinstance(lp, torch.Tensor) else torch.as_tensor(lp, dtype=torch.float32, device=self.device)
            for lp in self.log_probs]).to(self.device)

            values = torch.stack([
                v.squeeze() if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for v in self.values]).to(self.device)
        except RuntimeError as e:
            print(f"Tensor stacking error: {e}")
            print(f"log_probs types/shapes: {[ (type(lp), getattr(lp, 'shape', None)) for lp in self.log_probs ]}")
            print(f"values types/shapes: {[ (type(v), getattr(v, 'shape', None)) for v in self.values ]}")
            self.clear_episode_data()
            return None

        if log_probs.numel() == 0 or values.numel() == 0:
            self.clear_episode_data()
            return None

        # Compute returns (no grad)
        returns = torch.as_tensor(self.compute_discounted_rewards(), dtype=torch.float32, device=self.device)

        # Normalize returns for stability
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Match shapes
        returns = returns.view_as(values)

        # Advantages
        advantages = returns - values

        # Losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        total_loss = actor_loss + 0.5 * critic_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Log
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.total_losses.append(total_loss.item())

        # Clear buffers after update (on-policy)
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
        self.attacker_epsilon = []
        self.defender_epsilon = []
        
    def log_episode(self, episode_data: Dict, attacker_epsilon: Optional[float] = None, defender_epsilon: Optional[float] = None):
        """Log data from a completed episode."""
        self.attacker_rewards.append(episode_data.get('attacker_reward', 0))
        self.defender_rewards.append(episode_data.get('defender_reward', 0))
        self.episode_lengths.append(episode_data.get('length', 0))
        
        # Log epsilon values if provided
        if attacker_epsilon is not None:
            self.attacker_epsilon.append(attacker_epsilon)
        if defender_epsilon is not None:
            self.defender_epsilon.append(defender_epsilon)
        
        # Determine winner
        att_reward = episode_data.get('attacker_reward', 0)
        def_reward = episode_data.get('defender_reward', 0)

        if episode_data.get('winner') == 'attacker':
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
    
    def print_training_summary(self):
        """Print a summary of training metrics without plotting."""
        print("=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total Episodes: {len(self.attacker_rewards)}")
        
        if self.episode_lengths:
            print(f"Average Episode Length: {np.mean(self.episode_lengths):.2f} ± {np.std(self.episode_lengths):.2f}")
        
        print()
        print("REWARDS:")
        if self.attacker_rewards:
            print(f"  Attacker - Mean: {np.mean(self.attacker_rewards):.3f}, Std: {np.std(self.attacker_rewards):.3f}")
        if self.defender_rewards:
            print(f"  Defender - Mean: {np.mean(self.defender_rewards):.3f}, Std: {np.std(self.defender_rewards):.3f}")
        
        print()
        print("WIN RATES:")
        if self.attacker_wins:
            print(f"  Attacker: {np.mean(self.attacker_wins):.1%}")
        if self.defender_wins:
            print(f"  Defender: {np.mean(self.defender_wins):.1%}")
        
        # Recent performance (last 100 episodes)
        if len(self.attacker_rewards) >= 100:
            recent_att = self.attacker_rewards[-100:]
            recent_def = self.defender_rewards[-100:]
            recent_att_wins = self.attacker_wins[-100:]
            recent_def_wins = self.defender_wins[-100:]
            
            print()
            print("RECENT PERFORMANCE (Last 100 episodes):")
            print(f"  Attacker - Mean Reward: {np.mean(recent_att):.3f}")
            print(f"  Defender - Mean Reward: {np.mean(recent_def):.3f}")
            print(f"  Attacker Win Rate: {np.mean(recent_att_wins):.1%}")
            print(f"  Defender Win Rate: {np.mean(recent_def_wins):.1%}")
        
        print("=" * 60)
    
    def save_metrics(self, filepath: str):
        """Save training metrics to file."""
        metrics_data = {
            'attacker_rewards': self.attacker_rewards,
            'defender_rewards': self.defender_rewards,
            'episode_lengths': self.episode_lengths,
            'attacker_wins': self.attacker_wins,
            'defender_wins': self.defender_wins,
            'attacker_losses': self.attacker_losses,
            'defender_losses': self.defender_losses,
            'attacker_epsilon': self.attacker_epsilon,
            'defender_epsilon': self.defender_epsilon
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
        
        # Load epsilon values if available (for backward compatibility)
        self.attacker_epsilon = metrics_data.get('attacker_epsilon', [])
        self.defender_epsilon = metrics_data.get('defender_epsilon', [])
        
        print(f"Training metrics loaded from {filepath}")
