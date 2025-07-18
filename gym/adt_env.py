"""
Attack-Defense Tree Gymnasium Environment

A Gymnasium environment for Attack-Defense Trees based on JSON specification.
This environment supports alternating play between an attacker and defender,
where each player tries to optimize their strategy according to the tree structure.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from typing import Dict, Any, Tuple, Optional, List


class AttackDefenseTreeEnv(gym.Env):
    """
    A Gymnasium environment for Attack-Defense Trees based on JSON specification.
    
    This environment supports alternating play between an attacker and defender,
    where each player tries to optimize their strategy according to the tree structure.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, env_spec_file: str, render_mode: Optional[str] = None):
        """
        Initialize the environment from a JSON specification file.
        
        Args:
            env_spec_file (str): Path to the JSON environment specification
            render_mode (str): Rendering mode ('human' or None)
        """
        self.render_mode = render_mode
        
        # Load environment specification
        with open(env_spec_file, 'r') as f:
            self.env_spec = json.load(f)
        
        # Extract metadata
        self.goal = self.env_spec["metadata"]["goal"]
        self.actions_to_goal = set(self.env_spec["metadata"]["actions_to_goal"])
        
        # Set up state space
        self._setup_state_space()
        
        # Set up action spaces for both players
        self._setup_action_spaces()
        
        # Initialize state
        self.state: Dict[str, int] = {}
        self.reset()
    
    def _setup_state_space(self):
        """Set up the observation space based on the JSON specification."""
        self.state_vars = list(self.env_spec["state_space"].keys())
        self.state_size = len(self.state_vars)
        
        # Create state indices mapping
        self.state_indices = {var: i for i, var in enumerate(self.state_vars)}
        
        # Determine observation space bounds
        lows = []
        highs = []
        for var in self.state_vars:
            spec = self.env_spec["state_space"][var]
            lows.append(spec["low"])
            highs.append(spec["high"])
        
        self.observation_space = spaces.Box(
            low=np.array(lows, dtype=np.int32),
            high=np.array(highs, dtype=np.int32),
            dtype=np.int32
        )
    
    def _setup_action_spaces(self):
        """Set up action spaces for both players."""
        # Attacker actions
        self.attacker_actions = self.env_spec["action_space"]["attacker"]["actions"]
        self.num_attacker_actions = len(self.attacker_actions)
        print(f"Number of attacker actions: {self.num_attacker_actions}")
        
        # Defender actions  
        self.defender_actions = self.env_spec["action_space"]["defender"]["actions"]
        self.num_defender_actions = len(self.defender_actions)
        print(f"Number of defender actions: {self.num_defender_actions}")
        
        # Combined action space (will be filtered based on current player)
        max_actions = max(self.num_attacker_actions, self.num_defender_actions)
        self.action_space = spaces.Discrete(max_actions)
    
    def _get_state_vector(self) -> np.ndarray:
        """Convert current state dictionary to observation vector."""
        state_vector = np.zeros(self.state_size, dtype=np.int32)
        for var, value in self.state.items():
            if var in self.state_indices:
                state_vector[self.state_indices[var]] = value
        return state_vector
    
    def _check_preconditions(self, action_spec: Dict) -> bool:
        """Check if action preconditions are satisfied."""
        # Check if action can be performed based on the current state
        effect = action_spec.get("effect", None)
        if effect and effect in self.state:
            current_player = "attacker" if self.state["current_player"] == 0 else "defender"
            if current_player == "attacker" and self.state[effect] != 0:
                # Can't attack a node that's already compromised
                return False
            elif current_player == "defender" and self.state[effect] == 2:
                # Can't defend a node that's already protected
                return False

        conditions = action_spec["preconditions"]["conditions"]
        if not conditions:
            return True
        
        logic = action_spec["preconditions"]["logic"]
        required_values = action_spec["preconditions"]["required_values"]
        
        satisfied = []
        for condition in conditions:
            if condition in self.state:
                required_val = required_values.get(condition, 1)
                satisfied.append(self.state[condition] == required_val)
            else:
                satisfied.append(False)
        
        if logic == "AND":
            return all(satisfied)
        else:  # OR
            return any(satisfied) if satisfied else True
    
    def _apply_effects(self, action_spec: Dict, player: str) -> Dict[str, int]:
        """Apply action effects and return rewards."""
        rewards = {"attacker": 0, "defender": 0}
        
        # Get action effects from transitions
        if player == "attacker" and action_spec["name"] in self.env_spec["transitions"]["attacker"]:
            effects = self.env_spec["transitions"]["attacker"][action_spec["name"]]["effects"]
        elif player == "defender" and action_spec["name"] in self.env_spec["transitions"]["defender"]:
            effects = self.env_spec["transitions"]["defender"][action_spec["name"]]["effects"]
        else:
            effects = {}
        
        # Apply state changes
        for var, value in effects.items():
            if var in self.state:
                self.state[var] = value
        
        # Calculate rewards
        if player == "attacker":
            if action_spec["name"] in self.env_spec["rewards"]["attacker"]:
                rewards["attacker"] = self.env_spec["rewards"]["attacker"][action_spec["name"]]
        else:
            if action_spec["name"] in self.env_spec["rewards"]["defender"]:
                rewards["defender"] = self.env_spec["rewards"]["defender"][action_spec["name"]]
        
        # Terminal rewards
        if action_spec["name"] in self.env_spec["rewards"]["terminal_rewards"]:
            rewards[player] += self.env_spec["rewards"]["terminal_rewards"][action_spec["name"]]
        
        return rewards
    
    def _is_terminal(self) -> bool:
        """Check if the current state is terminal."""
        for terminal_spec in self.env_spec["terminal_states"]:
            condition = terminal_spec["condition"]
            # Simple evaluation for conditions like "DataExfiltration == 1"
            if "==" in condition:
                var, val = condition.split(" == ")
                if var.strip() in self.state:
                    return self.state[var.strip()] == int(val.strip())
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize state from JSON specification
        self.state = self.env_spec["initial_state"].copy()
        
        info = {
            "current_player": "attacker" if self.state["current_player"] == 0 else "defender",
            "goal": self.goal,
            "terminal": False
        }
        
        return self._get_state_vector(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        current_player = "attacker" if self.state["current_player"] == 0 else "defender"
        
        # Get action specification
        if current_player == "attacker":
            if action >= self.num_attacker_actions:
                action = self.num_attacker_actions - 1  # Default to wait
            action_spec = self.attacker_actions[str(action)]
        else:
            if action >= self.num_defender_actions:
                action = self.num_defender_actions - 1  # Default to wait
            action_spec = self.defender_actions[str(action)]
        
        # Check if action is valid (preconditions satisfied)
        valid_action = self._check_preconditions(action_spec)
        
        reward = 0
        if valid_action and action_spec["name"] != "wait":
            # Apply action effects
            rewards = self._apply_effects(action_spec, current_player)
            reward = rewards.get(current_player, 0)
        else:
            # Invalid action or wait - just switch turns
            next_player = 1 if current_player == "attacker" else 0
            self.state["current_player"] = next_player
        
        # Check if terminal state reached
        terminated = self._is_terminal()
        truncated = False  # Could add step limits here
        
        info = {
            "current_player": "attacker" if self.state["current_player"] == 0 else "defender",
            "action_taken": action_spec["name"],
            "action_valid": valid_action,
            "terminal": terminated
        }
        
        return self._get_state_vector(), reward, terminated, truncated, info
    
    def render(self):
        """Render the current state (simple text output)."""
        if self.render_mode == "human":
            current_player = "Attacker" if self.state["current_player"] == 0 else "Defender"
            print(f"\n=== Current Player: {current_player} ===")
            print(f"Goal ({self.goal}): {self.state[self.goal]}")
            
            # Show key state variables
            print("Key Attributes:")
            for var, value in self.state.items():
                if var not in ["current_player"] and not var.startswith("progress_"):
                    print(f"  {var}: {value}")
    
    def get_available_actions(self) -> List[int]:
        """Get list of valid actions for current player."""
        current_player = "attacker" if self.state["current_player"] == 0 else "defender"
        available = []
        
        if current_player == "attacker":
            actions = self.attacker_actions
        else:
            actions = self.defender_actions
        print(f"Available actions for {current_player}: {actions}")
        for action_id, action_spec in actions.items():
            if self._check_preconditions(action_spec):
                available.append(int(action_id))
        
        return available
