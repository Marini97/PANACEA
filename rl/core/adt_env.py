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
        self.goal_reached = False

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
                self.state['current_player'] = (self.state['current_player']+1) % 2 # switch player

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
        # Check if goal has been reached
        if self.state[self.goal] == 1:
            self.goal_reached = True
            return True
        
        # Check if agents have no available actions
        attacker_actions = self.get_available_actions("attacker")
        defender_actions = self.get_available_actions("defender")

        return len(attacker_actions) == 0 or len(defender_actions) == 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset goal tracking
        self.goal_reached = False
        
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
            return self._agent_selector.next()
        
        agent = self.agent_selection
        reward = 0
        action_spec = None
        valid_action = False
        action_name = ""
        
        if action is not None:
            if agent == "attacker":
                action_spec = self.attacker_actions[str(action)]
            else:
                action_spec = self.defender_actions[str(action)]
            action_name = action_spec["name"]
            # Apply action effects and get reward
            reward = self._apply_effects(action_spec, agent)
            self.rewards[agent] = reward
        
        # Update cumulative rewards
        for a in self.agents:
            self._cumulative_rewards[a] += self.rewards[a]
        
        # Check if terminal state reached
        terminated = self._is_terminal()
        
        if terminated:
            # Set termination for all agents
            self.terminations = {agent: True for agent in self.agents}
            
            # Assign terminal rewards based on who won
            if self.goal_reached and action_name is not "":
                # Attacker achieved goal - attacker wins
                terminal_reward = abs(self.env_spec["rewards"]["terminal_rewards"][action_name])
                self.rewards["attacker"] += 2000  # Positive reward for winning
                self.rewards["defender"] -= terminal_reward  # Penalty for losing
            else:
                # Defender prevented goal - defender wins
                self.rewards["defender"] += 2000  # Positive reward for winning
                self.rewards["attacker"] -= 0  # Penalty for losing

            # Update cumulative rewards with terminal rewards
            for a in self.agents:
                self._cumulative_rewards[a] += self.rewards[a]
        
        # Update observations
        obs = self._get_state_vector()
        self.observations = {agent: obs.copy() for agent in self.agents}
        
        # Get available actions for info
        available_actions = self.get_available_actions(agent)
        
        # Update infos
        self.infos[agent] = {
            "action_taken": action_name if action_spec is not None else "None",
            "available_actions_id": available_actions,
            "available_actions_name": [self.attacker_actions[str(a)]["name"] for a in available_actions] if agent == "attacker" else [self.defender_actions[str(a)]["name"] for a in available_actions],
            "terminal": terminated,
        }
        
        # Move to next agent
        self.agent_selection = self._agent_selector.next()
    
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
            if self._check_preconditions(action_spec, agent):
                available.append(int(action_id))
        
        return available
    
    def close(self):
        """Close the environment."""
        pass