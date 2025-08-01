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
        current_player = "attacker" if self.state["current_player"] == 0 else "defender"
        if effect and effect in self.state:
            if current_player == "attacker" and self.state[effect] != 0:
                # Can't attack a node that's already compromised or protected
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
        
        # Apply primary action effect
        if "effect" in action_spec and action_spec["effect"]:
            effect_var = action_spec["effect"]
            if effect_var in self.state:
                if player == "attacker":
                    # Attacker sets to 1 (compromised)
                    self.state[effect_var] = 1
                else:
                    # Defender sets to 2 (protected)
                    self.state[effect_var] = 2
        
        # Get additional action effects from transitions
        if player == "attacker" and action_spec["name"] in self.env_spec["transitions"]["attacker"]:
            effects = self.env_spec["transitions"]["attacker"][action_spec["name"]]["effects"]
        elif player == "defender" and action_spec["name"] in self.env_spec["transitions"]["defender"]:
            effects = self.env_spec["transitions"]["defender"][action_spec["name"]]["effects"]
        else:
            effects = {}
        
        # Apply additional state changes from transitions
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
            rewards["defender"] += self.env_spec["rewards"]["terminal_rewards"][action_spec["name"]]
        
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
        
        # Initialize all state variables to their minimum values
        self.state = {}
        for var, spec in self.env_spec["state_space"].items():
            self.state[var] = spec["low"]
        
        # Override with specific initial values from JSON
        for var, value in self.env_spec["initial_state"].items():
            self.state[var] = value
        
        info = {
            "current_player": "attacker" if self.state["current_player"] == 0 else "defender",
            "goal": self.goal,
            "terminal": False
        }
        
        return self._get_state_vector(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, Dict, bool, bool, Dict]:
        """Execute one step in the environment."""
        current_player = "attacker" if self.state["current_player"] == 0 else "defender"
        
        # Check if current player has any available actions (excluding wait)
        available_actions = self.get_available_actions()
        no_actions_available = len(available_actions) == 0
        
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
        
        winner = None
        rewards = {"attacker": 0, "defender": 0}
        
        if no_actions_available:
            # Current player has no valid actions - other player wins
            winner = "defender" if current_player == "attacker" else "attacker"
        elif valid_action and action_spec["name"] != "wait":
            # Apply action effects
            rewards = self._apply_effects(action_spec, current_player)
        else:
            # Invalid action or wait - just switch turns
            next_player = 1 if current_player == "attacker" else 0
            self.state["current_player"] = next_player
        
        # Check if terminal state reached
        terminated = self._is_terminal() or no_actions_available

        truncated = False  # Could add step limits here
        
        info = {
            "current_player": current_player,
            "action_taken": action_spec["name"],
            "action_valid": valid_action,
            "terminal": terminated,
            "winner": winner,
            "termination_reason": "no_actions" if no_actions_available else ("goal_achieved" if terminated else None),
        }
        
        return self._get_state_vector(), rewards, terminated, truncated, info
    
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
        # print(f"Available actions for {current_player}: {actions}")
        for action_id, action_spec in actions.items():
            # Exclude wait actions when checking for meaningful available actions
            if action_spec["name"] != "wait" and self._check_preconditions(action_spec):
                available.append(int(action_id))
        
        return available
