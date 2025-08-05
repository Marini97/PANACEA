from gymnasium import spaces
import numpy as np
import json
from typing import Dict, Optional, List
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers


class AttackDefenseTreeMultiAgentEnv(AECEnv):
    """
    A PettingZoo multi-agent environment for Attack-Defense Trees based on JSON specification.
    
    This environment supports alternating play between an attacker and defender,
    where each player tries to optimize their strategy.
    """
    
    metadata = {"render_modes": ["human"], "name": "attack_defense_tree_v1"}
    
    def __init__(self, env_spec_file: str, render_mode: Optional[str] = None):
        """
        Initialize the multi-agent environment from a JSON specification file.
        
        Args:
            env_spec_file (str): Path to the JSON environment specification
            render_mode (str): Rendering mode ('human' or None)
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
        
        # Set up state space
        self._setup_state_space()
        
        # Set up action spaces for both players
        self._setup_action_spaces()
        
        # Initialize agent selector for turn management
        self._agent_selector = agent_selector(self.agents)
        
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
        
        # Shared observation space for both agents
        obs_space = spaces.Box(
            low=np.array(lows, dtype=np.int32),
            high=np.array(highs, dtype=np.int32),
            dtype=np.int32
        )
        
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}
    
    def _setup_action_spaces(self):
        """Set up action spaces for both players."""
        # Attacker actions
        self.attacker_actions = self.env_spec["action_space"]["attacker"]["actions"]
        self.num_attacker_actions = len(self.attacker_actions)
        
        # Defender actions  
        self.defender_actions = self.env_spec["action_space"]["defender"]["actions"]
        self.num_defender_actions = len(self.defender_actions)
        
        # Define action spaces per agent
        self.action_spaces = {
            "attacker": spaces.Discrete(self.num_attacker_actions),
            "defender": spaces.Discrete(self.num_defender_actions)
        }
    
    def _get_state_vector(self) -> np.ndarray:
        """Convert current state dictionary to observation vector."""
        state_vector = np.zeros(self.state_size, dtype=np.int32)
        for var, value in self.state.items():
            if var in self.state_indices:
                state_vector[self.state_indices[var]] = value
        return state_vector
    
    def get_action_name(self, action: int, agent: str) -> str:
        """Get the name of the action for the specified agent."""
        if agent == "attacker":
            action_spec = self.attacker_actions.get(str(action), {})
        else:
            action_spec = self.defender_actions.get(str(action), {})
        
        return action_spec.get("name", "")
    
    def _check_preconditions(self, action_spec: Dict, agent: str) -> bool:
        """Check if action preconditions are satisfied."""
        # Check if action can be performed based on the current state
        effect = action_spec.get("effect", None)
        if effect and effect in self.state:
            if agent == "attacker" and self.state[effect] != 0:
                # Can't attack a node that's already compromised or protected
                return False
            elif agent == "defender" and self.state[effect] == 2:
                # Can't defend a node that's already protected
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
    
    def _apply_effects(self, action_spec: Dict, agent: str) -> int:
        """Apply action effects and return reward for the acting agent."""
        reward = 0
        
        # Apply primary action effect
        if "effect" in action_spec and action_spec["effect"]:
            effect_var = action_spec["effect"]
            if effect_var in self.state:
                if agent == "attacker":
                    # Attacker sets to 1 (compromised)
                    self.state[effect_var] = 1
                else:
                    # Defender sets to 2 (protected)
                    self.state[effect_var] = 2
        
        # Get additional action effects from transitions
        if agent == "attacker" and action_spec["name"] in self.env_spec["transitions"]["attacker"]:
            effects = self.env_spec["transitions"]["attacker"][action_spec["name"]]["effects"]
        elif agent == "defender" and action_spec["name"] in self.env_spec["transitions"]["defender"]:
            effects = self.env_spec["transitions"]["defender"][action_spec["name"]]["effects"]
        else:
            effects = {}
        
        # Apply additional state changes from transitions
        for var, value in effects.items():
            if var in self.state:
                self.state[var] = value
        
        # Calculate reward for the acting agent
        if agent == "attacker":
            if action_spec["name"] in self.env_spec["rewards"]["attacker"]:
                reward = int(self.env_spec["rewards"]["attacker"][action_spec["name"]])
        else:
            if action_spec["name"] in self.env_spec["rewards"]["defender"]:
                reward = int(self.env_spec["rewards"]["defender"][action_spec["name"]])
        
        return reward
    
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
        
        # Return observations and info
        return self.observations, self.infos
    
    def observe(self, agent: str) -> np.ndarray:
        """Return the observation for the specified agent."""
        return self.observations[agent] if agent in self.observations else np.zeros(self.state_size, dtype=np.int32)
    
    def step(self, action: int):
        """Execute one step in the environment for the current agent."""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            # If the agent is already terminated/truncated, return without doing anything
            return self._was_dead_step(action)
        
        agent = self.agent_selection
        valid_action = False
        action_spec = None
        reward = 0
        
        # Check if current agent has any available actions
        available_actions = self.get_available_actions(agent)
        no_actions_available = len(available_actions) == 0
        
        if not no_actions_available:
            if agent == "attacker":
                if action >= self.num_attacker_actions:
                    action = self.num_attacker_actions - 1  # Default to wait
                action_spec = self.attacker_actions[str(action)]
            else:
                if action >= self.num_defender_actions:
                    action = self.num_defender_actions - 1  # Default to wait
                action_spec = self.defender_actions[str(action)]
            # Check if action is valid (preconditions satisfied)
            valid_action = self._check_preconditions(action_spec, agent)
            
            if valid_action:
                # Apply action effects and get reward
                reward = self._apply_effects(action_spec, agent)
                self.rewards[agent] = reward
        
        # Update cumulative rewards
        for a in self.agents:
            self._cumulative_rewards[a] += self.rewards[a]
        
        # Check if terminal state reached
        terminated = self._is_terminal() or no_actions_available
        
        if terminated:
            # Set termination for all agents
            self.terminations = {agent: True for agent in self.agents}
            
            # Assign terminal rewards based on who won
            if self.state.get(self.goal, 0) == 1 and action_spec is not None:
                # Attacker achieved goal - attacker wins
                terminal_reward = abs(self.env_spec["rewards"]["terminal_rewards"][action_spec["name"]])
                self.rewards["attacker"] += 0  # Positive reward for winning
                self.rewards["defender"] -= terminal_reward  # Penalty for losing
            else:
                # Defender prevented goal - defender wins
                self.rewards["defender"] += 1000  # Positive reward for winning
                self.rewards["attacker"] -= 0  # Penalty for losing
            
            # Update cumulative rewards with terminal rewards
            for a in self.agents:
                self._cumulative_rewards[a] += self.rewards[a]
        
        # Update observations
        obs = self._get_state_vector()
        self.observations = {agent: obs.copy() for agent in self.agents}
        
        # Update infos
        self.infos[agent] = {
            "action_taken": action_spec["name"] if action_spec is not None else "None",
            "action_valid": valid_action,
            "available_actions": available_actions,
            "terminal": terminated,
        }
        
        # Move to next agent
        self.agent_selection = self._agent_selector.next()
    
    def _was_dead_step(self, action):
        """Handle step when agent is already dead/terminated."""
        # Do nothing, just advance to next agent
        self.agent_selection = self._agent_selector.next()
    
    def _get_opponent(self, agent: str) -> str:
        """Get the opponent of the given agent."""
        return "defender" if agent == "attacker" else "attacker"
    
    def render(self):
        """Render the current state (simple text output)."""
        if self.render_mode == "human":
            current_agent = self.agent_selection
            print(f"\n=== Current Agent: {current_agent.upper()} ===")
            print(f"Goal ({self.goal}): {self.state[self.goal]}")
            
            # Show key state variables
            print("Key Attributes:")
            for var, value in self.state.items():
                if var not in ["current_player"] and not var.startswith("progress_"):
                    print(f"  {var}: {value}")
            
            # Show available actions for current agent
            available = self.get_available_actions(current_agent)
            print(f"Available actions for {current_agent}: {available}")
    
    def get_available_actions(self, agent: str) -> List[int]:
        """Get list of valid actions for specified agent."""
        available = []
        
        if agent == "attacker":
            actions = self.attacker_actions
        else:
            actions = self.defender_actions
        
        for action_id, action_spec in actions.items():
            # Exclude wait actions when checking for meaningful available actions
            if action_spec["name"] != "wait" and self._check_preconditions(action_spec, agent):
                available.append(int(action_id))
        
        return available
    
    def close(self):
        """Close the environment."""
        pass