import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np

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
        nn.init.orthogonal_(self.actor[-1].weight)

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
            mask = torch.full_like(logits, -1e9)  # Use -1e9 instead of -inf for stability
            
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
        
        return action.item(), log_prob, value.squeeze()
    
class ADTAgent:
    """Actor-Critic agent for the ADT environment."""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 3e-4, 
                 gamma: float = 0.99, entropy_coef: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        # Networks
        self.network = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def get_action(self, state, available_actions=None):
        """Get action from the current policy."""
        # Ensure state is a numpy array and convert to tensor
        if isinstance(state, list):
            state = np.array(state)
        state_tensor = torch.FloatTensor(state)
        
        # Handle edge case where no actions are available
        if available_actions is None or len(available_actions) == 0:
            action = 0
            log_prob = torch.tensor(0.0)  # Zero log prob for random action
            value = torch.tensor(0.0)  # Zero value estimate
            return action, log_prob, value
        
        # Ensure available_actions is a proper list of integers
        if isinstance(available_actions, np.ndarray):
            available_actions = available_actions.tolist()
        elif not isinstance(available_actions, list):
            available_actions = list(available_actions)
        
        # Filter out invalid actions
        available_actions = [a for a in available_actions if 0 <= a < self.action_size]
        
        if len(available_actions) == 0:
            action = 0
            log_prob = torch.tensor(0.0)
            value = torch.tensor(0.0)
            return action, log_prob, value
        
        try:
            action, log_prob, value = self.network.get_action(state_tensor, available_actions)
            return action, log_prob, value
        except Exception as e:
            print(f"Error in get_action: {e}")
            # Fallback to random action from available actions
            action = np.random.choice(available_actions)
            log_prob = torch.tensor(0.0)
            value = torch.tensor(0.0)
            return action, log_prob, value
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store experience for training."""
        # Ensure consistent data types
        if isinstance(state, torch.Tensor):
            state = state.detach().numpy()
        if isinstance(state, list):
            state = np.array(state)
            
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns(self, next_value=0):
        """Compute discounted returns."""
        returns = []
        R = next_value
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
            
        return returns
    
    def update(self, next_value=0):
        """Update the network using collected experiences."""
        if len(self.states) == 0:
            return 0, 0, 0
            
        try:
            # Compute returns
            returns = self.compute_returns(next_value)
            
            # Convert to tensors with proper shape handling
            states = torch.FloatTensor(np.array(self.states))
            returns = torch.FloatTensor(returns)
            
            # Handle log_probs and values - ensure consistent shapes
            log_probs_list = []
            values_list = []
            
            for lp, v in zip(self.log_probs, self.values):
                # Convert to tensor and ensure scalar shape
                if not isinstance(lp, torch.Tensor):
                    lp = torch.tensor(float(lp))
                else:
                    lp = lp.clone().detach()
                
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(float(v))
                else:
                    v = v.clone().detach()
                
                # Ensure both are scalars (0-dimensional tensors)
                if lp.dim() > 0:
                    lp = lp.squeeze()
                if v.dim() > 0:
                    v = v.squeeze()
                
                # If they're still multi-dimensional, take the first element
                if lp.dim() > 0:
                    lp = lp.flatten()[0]
                if v.dim() > 0:
                    v = v.flatten()[0]
                
                log_probs_list.append(lp)
                values_list.append(v)
            
            # Stack tensors - they should all be scalars now
            log_probs = torch.stack(log_probs_list)
            values = torch.stack(values_list)
            
            # Compute advantages
            advantages = returns - values
            
            # Normalize advantages for stability
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute losses
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = F.mse_loss(values, returns)
            
            # Add entropy bonus for exploration
            logits, _ = self.network(states)
            policy = F.softmax(logits, dim=-1)
            entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=-1).mean()
            
            total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
            
            # Check for NaN in loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("Warning: NaN/Inf detected in loss, skipping update")
                self.clear_memory()
                return 0, 0, 0
            
            # Update network
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            
            self.optimizer.step()
            
            # Clear buffers
            self.clear_memory()
            
            return actor_loss.item(), critic_loss.item(), entropy.item()
            
        except Exception as e:
            print(f"Error in update: {e}")
            self.clear_memory()
            return 0, 0, 0
    
    def clear_memory(self):
        """Clear experience buffers."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()