from gymnasium import spaces
import numpy as np
import json
from typing import Dict, Optional, List, Tuple
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers


class AttackDefenseTreeMultiAgentTimeEnv(AECEnv):
    """
    A PettingZoo multi-agent environment for Attack-Defense Trees with time mechanics.
    
    This environment extends the basic ADT environment by incorporating time constraints
    where actions have execution durations and agents must wait for actions to complete.
    
    Key time mechanics:
    - Actions have time (duration to execute)
    - Agents enter progress states during action execution
    - Time passes through wait actions until completion
    """
    
    metadata = {"render_modes": ["human"], "name": "attack_defense_tree_time_v1"}
    
    def __init__(self, env_spec_file: str, render_mode: Optional[str] = None):
        """
        Initialize the time-based multi-agent environment from a JSON specification file.
        
        Args:
            env_spec_file (str): Path to the JSON environment specification
            render_mode (str): Rendering mode ('human' or None)
            max_time (int): Maximum time limit for episodes
        """
        super().__init__()
        
        self.render_mode = render_mode

        
        # Load environment specification
        with open(env_spec_file, 'r') as f:
            self.env_spec = json.load(f)
        
        # Extract metadata
        self.goal = self.env_spec["metadata"]["goal"]
        
        # Define agents
        self.possible_agents = ["attacker", "defender"]
        self.agents = self.possible_agents[:]
        
        # Extract action time costs first (needed for state space setup)
        self._extract_time_costs()
        
        # Set up state space (including time variables)
        self._setup_state_space()
        
        # Set up action spaces for both players
        self._setup_action_spaces()
        
        # Initialize agent selector for turn management
        self._agent_selector = agent_selector(self.agents)
        
        # Time tracking variables
        self.attacker_time_remaining = -1  # -1 means no action in progress
        self.defender_time_remaining = -1  # -1 means no action in progress
        
        # Single action tracking
        self.attacker_current_action = None  # Currently executing action spec
        self.defender_current_action = None  # Currently executing action spec
        
        # Last action completed by the attacker
        self.attacker_last_action = None
        
        # Initialize state
        self.state: Dict[str, int] = {}
        self.reset()
    
    def _extract_time_costs(self):
        """Extract time costs for all actions from the environment specification."""
        self.attacker_time_costs = {}
        self.defender_time_costs = {}
        
        # Add wait action time cost (always 1)
        self.attacker_time_costs["wait"] = 1
        self.defender_time_costs["wait"] = 1
        
        # Extract attacker action time costs from original actions
        for action_id, action_spec in self.env_spec["action_space"]["attacker"]["actions"].items():
            self.attacker_time_costs[action_spec["name"]] = action_spec.get("time", 1)
        
        # Extract defender action time costs from original actions
        for action_id, action_spec in self.env_spec["action_space"]["defender"]["actions"].items():
            self.defender_time_costs[action_spec["name"]] = action_spec.get("time", 1)
    
    def _setup_state_space(self):
        """Set up the observation space based on the JSON specification, including time variables."""
        # Get base state variables from JSON
        self.base_state_vars = list(self.env_spec["state_space"].keys())
        
        # Add time-related state variables
        self.time_state_vars = [
            "attacker_time_remaining", 
            "defender_time_remaining"
        ]
        
        # Combine all state variables
        self.state_vars = self.base_state_vars + self.time_state_vars
        self.state_size = len(self.state_vars)
        
        # Create state indices mapping
        self.state_indices = {var: i for i, var in enumerate(self.state_vars)}
        
        # Determine observation space bounds
        lows = []
        highs = []
        
        # Add bounds for base state variables
        for var in self.base_state_vars:
            spec = self.env_spec["state_space"][var]
            lows.append(spec["low"])
            highs.append(spec["high"])
        
        # Add bounds for time-related variables
        # Time remaining can be -1 (no action) to max possible time
        max_time = max([
            max(self.attacker_time_costs.values()) if self.attacker_time_costs else 1,
            max(self.defender_time_costs.values()) if self.defender_time_costs else 1
        ])
        for _ in self.time_state_vars:
            lows.append(-1)  # -1 means no action in progress
            highs.append(max_time)  # Maximum possible time cost

        # Shared observation space for both agents
        obs_space = spaces.Box(
            low=np.array(lows, dtype=np.int32),
            high=np.array(highs, dtype=np.int32),
            dtype=np.int32
        )
        
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}
    
    def _setup_action_spaces(self):
        """Set up action spaces for both players, including wait actions."""
        # Load original actions from environment specification
        original_attacker_actions = self.env_spec["action_space"]["attacker"]["actions"]
        original_defender_actions = self.env_spec["action_space"]["defender"]["actions"]
        
        # Create new action dictionaries with wait as action 0
        self.attacker_actions = {}
        self.defender_actions = {}
        
        # Add wait action as action 0 for attacker
        self.attacker_actions["0"] = {
            "name": "wait",
            "preconditions": {
                "conditions": [],
                "logic": "AND"
            },
            "effect": None,
            "cost": 0,
            "time": 1
        }
        
        # Add wait action as action 0 for defender
        self.defender_actions["0"] = {
            "name": "wait", 
            "preconditions": {
                "conditions": [],
                "logic": "AND"
            },
            "effect": None,
            "cost": 0,
            "time": 1
        }
        
        # Shift original actions by 1 (they start from action 1 now)
        for action_id, action_spec in original_attacker_actions.items():
            new_id = str(int(action_id) + 1)
            self.attacker_actions[new_id] = action_spec
            
        for action_id, action_spec in original_defender_actions.items():
            new_id = str(int(action_id) + 1)
            self.defender_actions[new_id] = action_spec
        
        # Update action counts
        self.num_attacker_actions = len(self.attacker_actions)
        self.num_defender_actions = len(self.defender_actions)
        
        # Define action spaces per agent
        self.action_spaces = {
            "attacker": spaces.Discrete(self.num_attacker_actions),
            "defender": spaces.Discrete(self.num_defender_actions)
        }
    
    def _get_state_vector(self) -> np.ndarray:
        """Convert current state dictionary to observation vector, including time variables."""
        state_vector = np.zeros(self.state_size, dtype=np.int32)
        
        # Set base state variables
        for var, value in self.state.items():
            if var in self.state_indices:
                state_vector[self.state_indices[var]] = value
        
        # Set time-related variables
        state_vector[self.state_indices["attacker_time_remaining"]] = self.attacker_time_remaining
        state_vector[self.state_indices["defender_time_remaining"]] = self.defender_time_remaining
        
        return state_vector
    
    def get_action_name(self, action: int, agent: str) -> str:
        """Get the name of the action for the specified agent."""
        if agent == "attacker":
            action_spec = self.attacker_actions.get(str(action), {})
        else:
            action_spec = self.defender_actions.get(str(action), {})
        
        return action_spec.get("name", "")
    
    def _check_preconditions(self, action_spec: Dict, agent: str) -> bool:
        """Check if action preconditions are satisfied and no action is in progress."""
        action_name = action_spec["name"]
        
        # Special handling for wait action
        if action_name == "wait":
            # Wait is only available when agent has an action in progress
            if agent == "attacker":
                return self.attacker_time_remaining > 0
            else:
                return self.defender_time_remaining > 0
        
        # Check if agent already has an action in progress (can't start new action)
        if agent == "attacker":
            if self.attacker_current_action is not None:
                return False
        else:
            if self.defender_current_action is not None:
                return False
        
        # Check basic preconditions (same as base environment)
        effect = action_spec.get("effect", None)
        if effect and effect in self.state:
            if agent == "attacker" and self.state[effect] != 0:
                return False
            elif agent == "defender" and self.state[effect] == 2:
                return False

        conditions = action_spec["preconditions"]["conditions"]
        if not conditions:
            return True
        
        logic = action_spec["preconditions"]["logic"]
        
        satisfied = []
        for condition in conditions:
            if condition in self.state:
                satisfied.append(self.state[condition] == 1)
            else:
                satisfied.append(False)
                
        # Evaluate based on logic (AND/OR)
        if logic == "AND":
            return all(satisfied)
        else:  # OR
            return any(satisfied) if satisfied else True
    
    def _start_action(self, action_spec: Dict, agent: str):
        """Start an action with time duration. Returns immediate reward."""
        action_name = action_spec["name"]
        
        if agent == "attacker":
            time_cost = self.attacker_time_costs.get(action_name, 1)
            self.attacker_time_remaining = time_cost
            self.attacker_current_action = action_spec
        else:
            time_cost = self.defender_time_costs.get(action_name, 1)
            self.defender_time_remaining = time_cost
            self.defender_current_action = action_spec
    
    def _check_completion_preconditions(self, action_spec: Dict, agent: str) -> bool:
        """Check if action preconditions are still satisfied for completion (ignore in-progress state)."""
        action_name = action_spec["name"]
        
        # Check basic preconditions (same as base environment)
        effect = action_spec.get("effect", None)
        if effect and effect in self.state:
            if agent == "attacker" and self.state[effect] != 0:
                return False
            elif agent == "defender" and self.state[effect] == 2:
                return False

        conditions = action_spec["preconditions"]["conditions"]
        if not conditions:
            return True
        
        logic = action_spec["preconditions"]["logic"]
        
        satisfied = []
        for condition in conditions:
            if condition in self.state:
                satisfied.append(self.state[condition] == 1)
            else:
                satisfied.append(False)
                
        # Evaluate based on logic (AND/OR)
        if logic == "AND":
            return all(satisfied)
        else:  # OR
            return any(satisfied) if satisfied else True

    def _complete_action(self, action_spec: Dict, agent: str) -> int:
        """Complete an action and apply its effects. Returns completion reward."""
        action_name = action_spec["name"]
        reward = 0
        
        # Re-check preconditions before applying effects (ignore in-progress state)
        if not self._check_completion_preconditions(action_spec, agent):
            # Action fails due to changed preconditions
            if agent == "attacker":
                self.attacker_current_action = None
                self.attacker_time_remaining = -1
            else:
                self.defender_current_action = None  
                self.defender_time_remaining = -1
            return -10  # Penalty for failed action
        
        # Mark action as completed
        if agent == "attacker":
            self.attacker_last_action = self.attacker_current_action
            self.attacker_current_action = None
            self.attacker_time_remaining = -1
        else:
            self.defender_current_action = None
            self.defender_time_remaining = -1
        
        # Apply primary action effect
        if "effect" in action_spec and action_spec["effect"]:
            effect_var = action_spec["effect"]
            if effect_var in self.state:
                if agent == "attacker":
                    self.state[effect_var] = 1  # Compromised
                else:
                    self.state[effect_var] = 2  # Protected
        
        # Get additional action effects from transitions
        if agent == "attacker" and action_name in self.env_spec["transitions"]["attacker"]:
            effects = self.env_spec["transitions"]["attacker"][action_name]["effects"]
        elif agent == "defender" and action_name in self.env_spec["transitions"]["defender"]:
            effects = self.env_spec["transitions"]["defender"][action_name]["effects"]
        else:
            effects = {}
        
        # Apply additional state changes from transitions
        for var, value in effects.items():
            if var in self.state:
                self.state[var] = value
        
        # Calculate reward for completing the action
        if agent == "attacker":
            if action_name in self.env_spec["rewards"]["attacker"]:
                reward = int(self.env_spec["rewards"]["attacker"][action_name])
        else:
            if action_name in self.env_spec["rewards"]["defender"]:
                reward = int(self.env_spec["rewards"]["defender"][action_name])
        
        return reward
    
    def _wait_action(self, agent: str) -> int:
        """Execute wait action - decrease time remaining and check for completion."""
        reward = 0
        
        if agent == "attacker" and self.attacker_time_remaining > 0:
            self.attacker_time_remaining -= 1
            
            # Check if action completes
            if self.attacker_time_remaining == 0 and self.attacker_current_action is not None:
                reward = self._complete_action(self.attacker_current_action, agent)
        
        elif agent == "defender" and self.defender_time_remaining > 0:
            self.defender_time_remaining -= 1
            
            # Check if action completes
            if self.defender_time_remaining == 0 and self.defender_current_action is not None:
                reward = self._complete_action(self.defender_current_action, agent)
        
        return reward
    
    def _is_terminal(self) -> bool:
        """Check if the current state is terminal."""
        # Check goal conditions
        for terminal_spec in self.env_spec["terminal_states"]:
            condition = terminal_spec["condition"]
            if "==" in condition:
                var, val = condition.split(" == ")
                if var.strip() in self.state:
                    return self.state[var.strip()] == int(val.strip())
        # Check if no actions are available for both agents
        attacker_actions = self.get_available_actions("attacker")
        defender_actions = self.get_available_actions("defender")
        return not attacker_actions or not defender_actions

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset agents
        self.agents = self.possible_agents[:]
        
        # Initialize all state variables to their minimum values
        self.state = {}
        for var, spec in self.env_spec["state_space"].items():
            self.state[var] = spec["low"]
        
        # Override with specific initial values from JSON
        for var, value in self.env_spec["initial_state"].items():
            self.state[var] = value
        
        # Reset time tracking
        self.attacker_time_remaining = -1
        self.defender_time_remaining = -1
        
        # Reset current actions
        self.attacker_current_action = None
        self.defender_current_action = None
        
        # Reset agent selector
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        # Initialize rewards, terminations, truncations, infos
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Initialize observations
        obs = self._get_state_vector()
        self.observations = {agent: obs.copy() for agent in self.agents}
        
        return self.observations, self.infos
    
    def observe(self, agent: str) -> np.ndarray:
        """Return the observation for the specified agent."""
        return self.observations[agent] if agent in self.observations else np.zeros(self.state_size, dtype=np.int32)

    def step(self, action: int | None):
        """Execute one step in the environment for the current agent."""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._agent_selector.next()
        
        agent = self.agent_selection
        reward = 0
        action_spec = None
        action_name = ""
        
        # Check if terminal state reached
        terminated = self._is_terminal()
        
        # Get action specification
        if action is not None:
            if agent == "attacker":
                if action >= self.num_attacker_actions:
                    action = self.num_attacker_actions - 1
                action_spec = self.attacker_actions[str(action)]
            else:
                if action >= self.num_defender_actions:
                    action = self.num_defender_actions - 1
                action_spec = self.defender_actions[str(action)]
            action_name = action_spec["name"]
        
            # Handle different action types
            if action_name == "wait":
                # Explicit wait action (action 0)
                reward = self._wait_action(agent)
            else:
                # Regular action - check if we can start it
                if self._check_preconditions(action_spec, agent):
                    self._start_action(action_spec, agent)
            
            if terminated:
                self.terminations = {agent: True for agent in self.agents}
                
                # Assign terminal rewards
                if self.state.get(self.goal, 0) == 1 and self.attacker_last_action:
                    # Attacker achieved goal
                    terminal_reward = abs(self.env_spec["rewards"]["terminal_rewards"][self.attacker_last_action['name']])
                    self.rewards["attacker"] += 0
                    self.rewards["defender"] -= terminal_reward
                else:
                    # Defender wins
                    self.rewards["defender"] += 1000
                    self.rewards["attacker"] -= 0
                
                # Update cumulative rewards with terminal rewards
                for a in self.agents:
                    self._cumulative_rewards[a] += self.rewards[a]
        
        # Update rewards
        self.rewards[agent] = reward
        for a in self.agents:
            self._cumulative_rewards[a] += self.rewards[a]
        
        # Update observations
        obs = self._get_state_vector()
        self.observations = {agent: obs.copy() for agent in self.agents}
        
        # Get available actions for info
        available_actions = self.get_available_actions(agent)
        
        # Update infos
        self.infos[agent] = {
            "action_taken": action_name,
            "available_actions": available_actions,
            "terminal": terminated,
            "attacker_time_remaining": self.attacker_time_remaining,
            "defender_time_remaining": self.defender_time_remaining
        }
        
        # Move to next agent
        self.agent_selection = self._agent_selector.next()
    
    def render(self):
        """Render the current state including time information."""
        if self.render_mode == "human":
            current_agent = self.agent_selection

            print(f"Goal ({self.goal}): {self.state[self.goal]}")
            
            # Show time remaining for each agent
            print(f"Attacker time remaining: {self.attacker_time_remaining}")
            print(f"Defender time remaining: {self.defender_time_remaining}")
            
            # Show actions in progress
            if self.attacker_current_action is not None:
                print(f"Attacker action in progress: {self.attacker_current_action['name']}")
            if self.defender_current_action is not None:
                print(f"Defender action in progress: {self.defender_current_action['name']}")
            
            # Show key state variables
            print("Key Attributes:")
            for var, value in self.state.items():
                if var not in ["current_player"] and not var.startswith("progress_"):
                    print(f"  {var}: {value}")
            
            # Show available actions for current agent
            available = self.get_available_actions(current_agent)
            print(f"Available actions for {current_agent}: {available}")
    
    def get_available_actions(self, agent: str) -> List[int]:
        """Get list of valid actions for specified agent, considering time constraints."""
        available = []
        
        if agent == "attacker":
            actions = self.attacker_actions
        else:
            actions = self.defender_actions
        
        for action_id, action_spec in actions.items():
            # Regular actions available if preconditions met and not in progress
            if self._check_preconditions(action_spec, agent):
                available.append(int(action_id))
        
        return available
    
    def get_action_progress(self, agent: str) -> Dict[str, bool]:
        """Get the progress status of current action for the specified agent."""
        if agent == "attacker":
            if self.attacker_current_action is not None:
                return {self.attacker_current_action["name"]: True}
            else:
                return {}
        else:
            if self.defender_current_action is not None:
                return {self.defender_current_action["name"]: True}
            else:
                return {}
    
    def get_time_remaining(self, agent: str) -> int:
        """Get the time remaining for the specified agent's current action."""
        if agent == "attacker":
            return self.attacker_time_remaining
        else:
            return self.defender_time_remaining
    
    def close(self):
        """Close the environment."""
        pass

