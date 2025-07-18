import json
import os
import argparse
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional
from tree import Tree
from tree_to_prism import parse_file, get_info


class TreeToGymEnvironment:
    """
    Converts a tree object into a Gymnasium environment specification.
    The environment is saved as JSON for later instantiation.
    """
    
    def __init__(self, tree: Tree, time_based: bool = False):
        self.tree = tree
        self.df = tree.to_dataframe()
        self.time_based = time_based
        self.goal, self.actions_to_goal, self.initial_attributes, \
        self.attacker_actions, self.defender_actions, \
        self.df_attacker, self.df_defender = get_info(self.df)
        
        # Initialize environment specification
        self.env_spec = {
            "metadata": {
                "goal": self.goal,
                "actions_to_goal": list(self.actions_to_goal),
                "initial_attributes": self.initial_attributes,
                "environment_type": "attack_defense_tree",
                "time_based": self.time_based
            },
            "state_space": {},
            "action_space": {},
            "transitions": {},
            "rewards": {},
            "initial_state": {},
            "terminal_states": []
        }
        
        self._build_environment()
    
    def _build_environment(self):
        """Build the complete environment specification."""
        self._define_state_space()
        self._define_action_space()
        self._define_initial_state()
        self._define_transitions()
        self._define_rewards()
        self._define_terminal_states()
    
    def _define_state_space(self):
        """Define the state space including all attributes and progress variables."""
        # Goal state
        self.env_spec["state_space"][self.goal] = {
            "type": "discrete",
            "low": 0,
            "high": 1,
            "description": "Goal achievement state"
        }
        
        # Current player turn
        self.env_spec["state_space"]["current_player"] = {
            "type": "discrete", 
            "low": 0,
            "high": 1,
            "description": "Current player (0=attacker, 1=defender)"
        }
        
        # Attacker attributes (can be 0=not achieved, 1=achieved, 2=defended)
        for attr in set(self.df_attacker.loc[self.df_attacker["Type"] == "Attribute"]["Label"].values):
            self.env_spec["state_space"][attr] = {
                "type": "discrete",
                "low": 0,
                "high": 2,
                "description": f"Attacker attribute: {attr}"
            }
        
        # Initial system attributes (start as 1=vulnerable, can be 2=protected)
        for attr in self.initial_attributes:
            self.env_spec["state_space"][attr] = {
                "type": "discrete",
                "low": 1,
                "high": 2,
                "description": f"Initial system attribute: {attr}"
            }
        
        # Defender attributes
        defender_attributes = set(self.df_defender.loc[self.df_defender["Type"] == "Attribute"]["Label"].values)
        for attr in defender_attributes:
            self.env_spec["state_space"][attr] = {
                "type": "discrete",
                "low": 0,
                "high": 1,
                "description": f"Defender attribute: {attr}"
            }
        
        # Time-based state variables
        if self.time_based:
            # Action progress states for attacker
            for action in self.attacker_actions.keys():
                self.env_spec["state_space"][f"progress_{action}"] = {
                    "type": "discrete",
                    "low": 0,
                    "high": 1,
                    "description": f"Progress state for attacker action: {action}"
                }
            
            # Action progress states for defender  
            for action in self.defender_actions.keys():
                self.env_spec["state_space"][f"progress_{action}"] = {
                    "type": "discrete",
                    "low": 0,
                    "high": 1,
                    "description": f"Progress state for defender action: {action}"
                }
            # Get maximum time values for bounds
            attacker_max_time = max([int(data["time"]) if data["time"] else 0 
                                   for data in self.attacker_actions.values()] + [0])
            defender_max_time = max([int(data["time"]) if data["time"] else 0 
                                   for data in self.defender_actions.values()] + [0])
            
            # Time counters for each player
            self.env_spec["state_space"]["time_attacker"] = {
                "type": "discrete",
                "low": -1,
                "high": attacker_max_time,
                "description": "Time counter for attacker (-1 means can act immediately)"
            }
            
            self.env_spec["state_space"]["time_defender"] = {
                "type": "discrete", 
                "low": -1,
                "high": defender_max_time,
                "description": "Time counter for defender (-1 means can act immediately)"
            }
    
    def _define_action_space(self):
        """Define available actions for both players."""
        self.env_spec["action_space"]["attacker"] = {
            "type": "discrete",
            "actions": {}
        }
        
        self.env_spec["action_space"]["defender"] = {
            "type": "discrete", 
            "actions": {}
        }
        
        if self.time_based:
            self._define_time_based_action_space()
        else:
            self._define_standard_action_space()
    
    def _define_standard_action_space(self):
        """Define action space for non-time-based environment."""
        # Add attacker actions
        action_id = 0
        for action_name, action_data in self.attacker_actions.items():
            cost = int(action_data["cost"]) if action_data["cost"] else 0
            
            # Build conditions list - only include the original preconditions
            conditions = action_data["preconditions"].copy()
            
            # Determine logic based on refinement type
            logic = "OR" if action_data["refinement"] == "disjunctive" else "AND"
            
            self.env_spec["action_space"]["attacker"]["actions"][action_id] = {
                "name": action_name,
                "preconditions": {
                    "conditions": conditions,
                    "logic": logic,
                    "required_values": {cond: 1 for cond in action_data["preconditions"]}
                },
                "effect": action_data["effect"],
                "cost": cost,
                "refinement": action_data["refinement"]
            }
            action_id += 1
        
        # Add defender actions
        action_id = 0
        for action_name, action_data in self.defender_actions.items():
            cost = int(action_data["cost"]) if action_data["cost"] else 0
            
            # Build conditions list - only include the original preconditions
            conditions = action_data["preconditions"].copy()
            
            self.env_spec["action_space"]["defender"]["actions"][action_id] = {
                "name": action_name,
                "preconditions": {
                    "conditions": conditions,
                    "logic": "AND",  # Defender actions are typically conjunctive
                    "required_values": {cond: 1 for cond in action_data["preconditions"]}
                },
                "effect": action_data["effect"],
                "cost": cost,
                "refinement": action_data["refinement"]
            }
            action_id += 1
    
    def _define_time_based_action_space(self):
        """Define action space for time-based environment."""
        # Attacker actions: start, end, fail, wait
        action_id = 0
        for action_name, action_data in self.attacker_actions.items():
            # Start
            self.env_spec["action_space"]["attacker"]["actions"][action_id] = {
                "name": f"start_{action_name}",
                "preconditions": {
                    "conditions": action_data["preconditions"] + [f"time_attacker < 0", f"progress_{action_name} == 0"],
                    "logic": "AND",  # All preconditions must be met to start an action
                    "required_values": {cond: 1 for cond in action_data["preconditions"]}
                },
                "effect": None,
                "cost": int(action_data["cost"]) if action_data["cost"] else 0,
                "time": int(action_data["time"]) if action_data["time"] else 0,
                "refinement": action_data["refinement"],
                "action_type": "start"
            }
            action_id += 1
            # End
            self.env_spec["action_space"]["attacker"]["actions"][action_id] = {
                "name": f"end_{action_name}",
                "preconditions": {
                    "conditions": action_data["preconditions"] + [f"time_attacker == 0", f"progress_{action_name} == 1"],
                    "logic": "AND",  # All preconditions must be met to end an action
                    "required_values": {cond: 1 for cond in action_data["preconditions"]}
                },
                "effect": action_data["effect"],
                "cost": 0,
                "time": 0,
                "refinement": action_data["refinement"],
                "action_type": "end"
            }
            action_id += 1
            # Fail - structured the same way as transitions
            fail_conditions = []
            if action_data["preconditions"]:
                if action_data["refinement"] == "disjunctive":
                    # For OR, fail if ALL preconditions are false
                    fail_conditions.append(" & ".join([f"{cond} != 1" for cond in action_data["preconditions"]]))
                else:
                    # For AND, fail if ANY precondition is false
                    fail_conditions.extend([f"{cond} != 1" for cond in action_data["preconditions"]])
            
            # Create the precondition: (time == 0 AND progress == 1) AND (at least one failure condition)
            base_conditions = [f"time_attacker == 0", f"progress_{action_name} == 1"]
            if fail_conditions:
                # Add the failure conditions as a single OR clause
                failure_clause = " | ".join(fail_conditions)
                base_conditions.append(failure_clause)
            
            self.env_spec["action_space"]["attacker"]["actions"][action_id] = {
                "name": f"fail_{action_name}",
                "preconditions": {
                    "conditions": base_conditions,
                    "logic": "AND",
                    "required_values": {}
                },
                "effect": None,
                "cost": 0,
                "time": 0,
                "refinement": action_data["refinement"],
                "action_type": "fail"
            }
            action_id += 1
        # Wait
        self.env_spec["action_space"]["attacker"]["actions"][action_id] = {
            "name": "wait_attacker",
            "preconditions": {
                "conditions": ["time_attacker > 0"],
                "logic": "AND",
                "required_values": {}
            },
            "effect": None,
            "cost": 0,
            "time": 0,
            "refinement": None,
            "action_type": "wait"
        }
        
        # Defender actions: start, end, fail, wait
        action_id = 0
        for action_name, action_data in self.defender_actions.items():
            # Start
            self.env_spec["action_space"]["defender"]["actions"][action_id] = {
                "name": f"start_{action_name}",
                "preconditions": {
                    "conditions": action_data["preconditions"] + [f"time_defender < 0", f"progress_{action_name} == 0"],
                    "logic": "AND",  # All preconditions must be met to start an action
                    "required_values": {cond: 1 for cond in action_data["preconditions"]}
                },
                "effect": None,
                "cost": int(action_data["cost"]) if action_data["cost"] else 0,
                "time": int(action_data["time"]) if action_data["time"] else 0,
                "refinement": action_data["refinement"],
                "action_type": "start"
            }
            action_id += 1
            # End
            self.env_spec["action_space"]["defender"]["actions"][action_id] = {
                "name": f"end_{action_name}",
                "preconditions": {
                    "conditions": action_data["preconditions"] + [f"time_defender == 0", f"progress_{action_name} == 1"],
                    "logic": "AND",  # All preconditions must be met to end an action
                    "required_values": {cond: 1 for cond in action_data["preconditions"]}
                },
                "effect": action_data["effect"],
                "cost": 0,
                "time": 0,
                "refinement": action_data["refinement"],
                "action_type": "end"
            }
            action_id += 1
            # Fail - structured the same way as transitions
            fail_conditions = []
            if action_data["preconditions"]:
                if action_data["refinement"] == "disjunctive":
                    fail_conditions.append(" & ".join([f"{cond} != 1" for cond in action_data["preconditions"]]))
                else:
                    fail_conditions.extend([f"{cond} != 1" for cond in action_data["preconditions"]])
            
            # Create the precondition: (time == 0 AND progress == 1) AND (at least one failure condition)
            base_conditions = [f"time_defender == 0", f"progress_{action_name} == 1"]
            if fail_conditions:
                # Add the failure conditions as a single OR clause
                failure_clause = " | ".join(fail_conditions)
                base_conditions.append(failure_clause)
            
            self.env_spec["action_space"]["defender"]["actions"][action_id] = {
                "name": f"fail_{action_name}",
                "preconditions": {
                    "conditions": base_conditions,
                    "logic": "AND",
                    "required_values": {}
                },
                "effect": None,
                "cost": 0,
                "time": 0,
                "refinement": action_data["refinement"],
                "action_type": "fail"
            }
            action_id += 1
        # Wait
        self.env_spec["action_space"]["defender"]["actions"][action_id] = {
            "name": "wait_defender",
            "preconditions": {
                "conditions": ["time_defender > 0"],
                "logic": "AND",
                "required_values": {}
            },
            "effect": None,
            "cost": 0,
            "time": 0,
            "refinement": None,
            "action_type": "wait"
        }

    def _define_initial_state(self):
        """Define the initial state of the environment."""
        # Goal not achieved initially
        self.env_spec["initial_state"][self.goal] = 0
        
        # Start with attacker turn
        self.env_spec["initial_state"]["current_player"] = 0
        
        # All attacker attributes start as not achieved
        for attr in set(self.df_attacker.loc[self.df_attacker["Type"] == "Attribute"]["Label"].values):
            self.env_spec["initial_state"][attr] = 0
        
        # Initial system attributes start as vulnerable
        for attr in self.initial_attributes:
            self.env_spec["initial_state"][attr] = 1
        
        # All defender attributes start as not achieved
        defender_attributes = set(self.df_defender.loc[self.df_defender["Type"] == "Attribute"]["Label"].values)
        for attr in defender_attributes:
            self.env_spec["initial_state"][attr] = 0
        
        # Time-based initial states
        if self.time_based:
            # All action progress states start as not in progress
            for action in self.attacker_actions.keys():
                self.env_spec["initial_state"][f"progress_{action}"] = 0
            
            for action in self.defender_actions.keys():
                self.env_spec["initial_state"][f"progress_{action}"] = 0
                
            self.env_spec["initial_state"]["time_attacker"] = -1  # Can act immediately
            self.env_spec["initial_state"]["time_defender"] = -1  # Can act immediately
    
    def _define_transitions(self):
        """Define state transitions for each action."""
        self.env_spec["transitions"]["attacker"] = {}
        self.env_spec["transitions"]["defender"] = {}
        
        if self.time_based:
            self._define_time_based_transitions()
        else:
            self._define_standard_transitions()
    
    def _define_standard_transitions(self):
        """Define transitions for non-time-based environment."""
        # Attacker transitions
        for action_name, action_data in self.attacker_actions.items():
            self.env_spec["transitions"]["attacker"][action_name] = {
                "preconditions": self._format_preconditions(action_data),
                "effects": self._format_effects(action_data, "attacker"),
            }
        
        # Defender transitions
        for action_name, action_data in self.defender_actions.items():
            self.env_spec["transitions"]["defender"][action_name] = {
                "preconditions": self._format_preconditions(action_data),
                "effects": self._format_effects(action_data, "defender"),
            }
    
    def _define_time_based_transitions(self):
        """Define transitions for time-based environment following PRISM logic."""
        # Attacker transitions
        for action_name, action_data in self.attacker_actions.items():
            time = int(action_data["time"]) if action_data["time"] else 0
            
            # Start action - can only start if time < 0, not in progress, and preconditions met
            self.env_spec["transitions"]["attacker"][f"start_{action_name}"] = {
                "preconditions": {
                    "conditions": action_data["preconditions"] + [f"time_attacker < 0", f"progress_{action_name} == 0"],
                    "logic": "AND"  # All preconditions must be met to start an action
                },
                "effects": {
                    f"progress_{action_name}": 1,
                    "time_attacker": time,
                    "current_player": 1
                },
                "failure_conditions": {
                    "conditions": [f"time_attacker >= 0", f"progress_{action_name} != 0"] + [f"{cond} != 1" for cond in action_data["preconditions"]],
                    "logic": "OR"
                }
            }
            
            # End action - can only end if time == 0, in progress, and preconditions still met
            self.env_spec["transitions"]["attacker"][f"end_{action_name}"] = {
                "preconditions": {
                    "conditions": action_data["preconditions"] + [f"time_attacker == 0", f"progress_{action_name} == 1"],
                    "logic": "AND"  # All preconditions must be met to end an action
                },
                "effects": self._format_time_effects(action_data, "attacker", action_name),
                "failure_conditions": {
                    "conditions": [f"time_attacker != 0", f"progress_{action_name} != 1"] + [f"{cond} != 1" for cond in action_data["preconditions"]],
                    "logic": "OR"
                }
            }
            
            # Fail action - can fail if time == 0, in progress, AND (preconditions no longer met OR effect already achieved)
            fail_conditions = []
            if action_data["preconditions"]:
                if action_data["refinement"] == "disjunctive":
                    # For OR refinement, fail if ALL preconditions are false
                    fail_conditions.append(" & ".join([f"{cond} != 1" for cond in action_data["preconditions"]]))
                else:
                    # For AND refinement, fail if ANY precondition is false
                    fail_conditions.extend([f"{cond} != 1" for cond in action_data["preconditions"]])
            
            # Also fail if effect is already achieved
            fail_conditions.append(f"{action_data['effect']} != 0")
            
            # Create the precondition: (time == 0 AND progress == 1) AND (at least one failure condition)
            base_conditions = [f"time_attacker == 0", f"progress_{action_name} == 1"]
            if fail_conditions:
                # Add the failure conditions as a single OR clause
                failure_clause = " | ".join(fail_conditions)
                base_conditions.append(failure_clause)
            
            self.env_spec["transitions"]["attacker"][f"fail_{action_name}"] = {
                "preconditions": {
                    "conditions": base_conditions,
                    "logic": "AND"
                },
                "effects": {
                    f"progress_{action_name}": 0,
                    "time_attacker": -1,
                    "current_player": 1
                },
                "failure_conditions": {
                    "conditions": [f"time_attacker != 0", f"progress_{action_name} != 1"],
                    "logic": "OR"
                }
            }

        # Wait attacker - can only wait if time > 0
        self.env_spec["transitions"]["attacker"]["wait_attacker"] = {
            "preconditions": {
                "conditions": ["time_attacker > 0"],
                "logic": "AND"
            },
            "effects": {
                "time_attacker": "time_attacker - 1",
                "current_player": 1
            },
            "failure_conditions": {
                "conditions": ["time_attacker <= 0"],
                "logic": "OR"
            }
        }
        
        # Defender transitions
        for action_name, action_data in self.defender_actions.items():
            time = int(action_data["time"]) if action_data["time"] else 0
            
            # Start action
            self.env_spec["transitions"]["defender"][f"start_{action_name}"] = {
                "preconditions": {
                    "conditions": action_data["preconditions"] + [f"time_defender < 0", f"progress_{action_name} == 0"],
                    "logic": "AND"  # All preconditions must be met to start an action
                },
                "effects": {
                    f"progress_{action_name}": 1,
                    "time_defender": time,
                    "current_player": 0
                },
                "failure_conditions": {
                    "conditions": [f"time_defender >= 0", f"progress_{action_name} != 0"] + [f"{cond} != 1" for cond in action_data["preconditions"]],
                    "logic": "OR"
                }
            }
            
            # End action
            self.env_spec["transitions"]["defender"][f"end_{action_name}"] = {
                "preconditions": {
                    "conditions": action_data["preconditions"] + [f"time_defender == 0", f"progress_{action_name} == 1"],
                    "logic": "AND"  # All preconditions must be met to end an action
                },
                "effects": self._format_time_effects(action_data, "defender", action_name),
                "failure_conditions": {
                    "conditions": [f"time_defender != 0", f"progress_{action_name} != 1"] + [f"{cond} != 1" for cond in action_data["preconditions"]],
                    "logic": "OR"
                }
            }
            
            # Fail action
            fail_conditions = []
            if action_data["preconditions"]:
                if action_data["refinement"] == "disjunctive":
                    fail_conditions.append(" & ".join([f"{cond} != 1" for cond in action_data["preconditions"]]))
                else:
                    fail_conditions.extend([f"{cond} != 1" for cond in action_data["preconditions"]])
            
            # Create the precondition: (time == 0 AND progress == 1) AND (at least one failure condition)
            base_conditions = [f"time_defender == 0", f"progress_{action_name} == 1"]
            if fail_conditions:
                # Add the failure conditions as a single OR clause
                failure_clause = " | ".join(fail_conditions)
                base_conditions.append(failure_clause)
            
            self.env_spec["transitions"]["defender"][f"fail_{action_name}"] = {
                "preconditions": {
                    "conditions": base_conditions,
                    "logic": "AND"
                },
                "effects": {
                    f"progress_{action_name}": 0,
                    "time_defender": -1,
                    "current_player": 0
                },
                "failure_conditions": {
                    "conditions": [f"time_defender != 0", f"progress_{action_name} != 1"],
                    "logic": "OR"
                }
            }

        # Wait defender - can only wait if time > 0
        self.env_spec["transitions"]["defender"]["wait_defender"] = {
            "preconditions": {
                "conditions": ["time_defender > 0"],
                "logic": "AND"
            },
            "effects": {
                "time_defender": "time_defender - 1",
                "current_player": 0
            },
            "failure_conditions": {
                "conditions": ["time_defender <= 0"],
                "logic": "OR"
            }
        }

    def _format_precondition_check(self, preconditions, refinement):
        """Build precondition check string."""
        if not preconditions:
            return ""
        
        if refinement == "disjunctive":
            return " OR ".join([f"{p} == 1" for p in preconditions])
        else:
            return " AND ".join([f"{p} == 1" for p in preconditions])
    
    def _build_fail_check(self, preconditions, refinement):
        """Build failure condition check string."""
        if not preconditions:
            return ""
        
        if refinement == "disjunctive":
            # For OR, fails if ALL preconditions are false
            return " AND ".join([f"{p} != 1" for p in preconditions])
        else:
            # For AND, fails if ANY precondition is false
            return " OR ".join([f"{p} != 1" for p in preconditions])
    
    def _format_preconditions(self, action_data: Dict) -> Dict:
        """Format preconditions for an action."""
        preconditions = action_data["preconditions"]
        refinement = action_data["refinement"]
        
        # Build conditions list - only include the original preconditions
        conditions = preconditions.copy()
        
        # Determine logic based on refinement type
        logic = "OR" if refinement == "disjunctive" else "AND"
        
        return {
            "conditions": conditions,
            "logic": logic,
            "required_values": {cond: 1 for cond in preconditions}
        }
    
    def _format_effects(self, action_data: Dict, player: str) -> Dict:
        """Format effects of successful action execution."""
        effect = action_data["effect"]
        effects = {}
        
        if effect:
            if player == "attacker":
                effects[effect] = 1
            else:
                # Defender effects can either set attribute to 1 or defend (set to 2)
                defender_attributes = set(self.df_defender.loc[self.df_defender["Type"] == "Attribute"]["Label"].values)
                if effect in defender_attributes:
                    effects[effect] = 1
                else:
                    effects[effect] = 2
        
        # Switch turn
        effects["current_player"] = 1 if player == "attacker" else 0
        
        return effects
    
    def _format_failure_conditions(self, action_data: Dict, player: str) -> Dict:
        """Format conditions under which an action fails."""
        preconditions = action_data["preconditions"]
        refinement = action_data["refinement"]
        
        failure_conditions = []
        
        # Action fails if preconditions are not met
        if preconditions:
            if refinement == "disjunctive":
                # For OR, fails if ALL preconditions are false
                failure_conditions.append(" AND ".join([f"{cond} != 1" for cond in preconditions]))
            else:
                # For AND, fails if ANY precondition is false
                failure_conditions.extend([f"{cond} != 1" for cond in preconditions])
        
        return {
            "conditions": failure_conditions,
            "logic": "OR"
        }
    
    def _format_time_preconditions(self, action_data: Dict, player: str, action_type: str) -> Dict:
        """Format preconditions for time-based actions."""
        preconditions = action_data["preconditions"]
        refinement = action_data["refinement"]
        
        # Find action name by matching effect
        effect = action_data["effect"]
        action_name = None
        actions_dict = self.attacker_actions if player == "attacker" else self.defender_actions
        for name, data in actions_dict.items():
            if data["effect"] == effect:
                action_name = name
                break
        
        conditions = []
        
        if action_type == "start":
            conditions.append(f"time_{player} < 0")
            if action_name:
                conditions.append(f"progress_{action_name} == 0")
            conditions.extend(preconditions)
        elif action_type == "end":
            conditions.append(f"time_{player} == 0")
            if action_name:
                conditions.append(f"progress_{action_name} == 1")
            conditions.extend(preconditions)
        
        return {
            "conditions": conditions,
            "logic": "OR" if refinement == "disjunctive" else "AND",
            "required_values": {cond: 1 for cond in preconditions}
        }
    
    def _format_time_effects(self, action_data: Dict, player: str, action_name: str) -> Dict:
        effect = action_data["effect"]
        effects = {}
        if effect:
            if player == "attacker":
                effects[effect] = 1
            else:
                defender_attributes = set(self.df_defender.loc[self.df_defender["Type"] == "Attribute"]["Label"].values)
                if effect in defender_attributes:
                    effects[effect] = 1
                else:
                    effects[effect] = 2
        effects[f"progress_{action_name}"] = 0
        effects[f"time_{player}"] = -1
        effects["current_player"] = 1 if player == "attacker" else 0
        return effects
    
    def _format_time_failure_conditions(self, action_data: Dict, player: str, action_type: str) -> Dict:
        """Format failure conditions for time-based actions."""
        preconditions = action_data["preconditions"]
        refinement = action_data["refinement"]
        effect = action_data["effect"]
        
        failure_conditions = []
        
        # Find action name by effect
        actions_dict = self.attacker_actions if player == "attacker" else self.defender_actions
        action_name = None
        for name, data in actions_dict.items():
            if data["effect"] == effect:
                action_name = name
                break
        
        if action_type == "start":
            # Fails if already in progress or time is not ready
            failure_conditions.append(f"time_{player} >= 0")
            if action_name:
                failure_conditions.append(f"progress_{action_name} != 0")
        elif action_type == "end":
            # Fails if not ready to complete or time not elapsed
            failure_conditions.append(f"time_{player} != 0")
            if action_name:
                failure_conditions.append(f"progress_{action_name} != 1")
        
        # Action fails if preconditions are not met
        if preconditions:
            if refinement == "disjunctive":
                # For OR, fails if ALL preconditions are false
                failure_conditions.append(" AND ".join([f"{cond} != 1" for cond in preconditions]))
            else:
                # For AND, fails if ANY precondition is false
                failure_conditions.extend([f"{cond} != 1" for cond in preconditions])
        
        return {
            "conditions": failure_conditions,
            "logic": "OR"
        }

    def _define_rewards(self):
        """Define reward structure for both players."""
        self.env_spec["rewards"] = {
            "attacker": {},
            "defender": {},
            "terminal_rewards": {}
        }
        
        if self.time_based:
            # Time-based rewards - costs are paid when starting actions
            for action_name, action_data in self.attacker_actions.items():
                cost = int(action_data["cost"]) if action_data["cost"] else 0
                self.env_spec["rewards"]["attacker"][f"start_{action_name}"] = -cost
                self.env_spec["rewards"]["attacker"][f"end_{action_name}"] = 0
                self.env_spec["rewards"]["attacker"][f"fail_{action_name}"] = 0
            self.env_spec["rewards"]["attacker"]["wait_attacker"] = 0
            for action_name, action_data in self.defender_actions.items():
                cost = int(action_data["cost"]) if action_data["cost"] else 0
                self.env_spec["rewards"]["defender"][f"start_{action_name}"] = -cost
                self.env_spec["rewards"]["defender"][f"end_{action_name}"] = 0
                self.env_spec["rewards"]["defender"][f"fail_{action_name}"] = 0
            self.env_spec["rewards"]["defender"]["wait_defender"] = 0
            # Terminal rewards for goal-reaching actions
            for action in self.actions_to_goal:
                if action in self.attacker_actions:
                    cost = int(self.attacker_actions[action]["cost"]) if self.attacker_actions[action]["cost"] else 0
                    self.env_spec["rewards"]["terminal_rewards"][f"end_{action}"] = -cost * 10
        else:
            # Standard rewards
            for action_name, action_data in self.attacker_actions.items():
                cost = int(action_data["cost"]) if action_data["cost"] else 0
                self.env_spec["rewards"]["attacker"][action_name] = -cost  # Negative because it's a cost
            
            for action_name, action_data in self.defender_actions.items():
                cost = int(action_data["cost"]) if action_data["cost"] else 0
                self.env_spec["rewards"]["defender"][action_name] = -cost  # Negative because it's a cost
            
            # Terminal rewards
            for action in self.actions_to_goal:
                if action in self.attacker_actions:
                    cost = int(self.attacker_actions[action]["cost"]) if self.attacker_actions[action]["cost"] else 0
                    self.env_spec["rewards"]["terminal_rewards"][action] = -cost * 10
    
    def _define_terminal_states(self):
        """Define terminal state conditions."""
        self.env_spec["terminal_states"] = [
            {
                "condition": f"{self.goal} == 1",
                "description": "Goal achieved by attacker"
            }
        ]
    
    def save_environment(self, filename: str):
        """Save the environment specification to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.env_spec, f, indent=2)
        print(f"Environment saved to {filename}")
    
    def get_environment_spec(self) -> Dict:
        """Return the environment specification."""
        return self.env_spec


def convert_tree_to_gym_env(xml_file: str, output_file: Optional[str] = None, time_based: bool = False):
    """
    Convert an XML tree file to a Gymnasium environment specification.
    
    Args:
        xml_file (str): Path to the XML tree file
        output_file (str): Path to save the JSON environment specification
                          If None, uses the XML filename with .json extension
        time_based (bool): Whether to create a time-based environment
    
    Returns:
        Dict: The environment specification
    """
    # Parse the XML file
    tree = parse_file(xml_file)
    
    # Create environment converter
    env_converter = TreeToGymEnvironment(tree, time_based)
    
    # Generate output filename if not provided
    if output_file is None:
        suffix = '_time_env.json' if time_based else '_env.json'
        output_file = xml_file.replace('.xml', suffix)
    
    # Save environment
    env_converter.save_environment(output_file)
    
    return env_converter.get_environment_spec()


def load_environment_spec(json_file: str) -> Dict:
    """
    Load an environment specification from a JSON file.
    
    Args:
        json_file (str): Path to the JSON environment file
    
    Returns:
        Dict: The environment specification
    """
    with open(json_file, 'r') as f:
        env_spec = json.load(f)
    return env_spec


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Convert XML Attack-Defense Tree to Gymnasium environment specification')
    parser.add_argument('--input', '-i', type=str, help='Path to the XML file from ADTool', required=True)
    parser.add_argument('--output', '-o', type=str, help='Path to the output JSON environment file')
    parser.add_argument('--prune', '-p', type=str, help='Name of the subtree root to keep')
    parser.add_argument('--time', '-t', action='store_true', help='Generate a time-based environment')
    args = parser.parse_args()
    
    # Generate output filename if not provided or if directory is provided
    if args.output is None:
        suffix = '_time_env.json' if args.time else '_env.json'
        args.output = args.input.replace('.xml', suffix)
    elif os.path.isdir(args.output):
        # If output is a directory, generate filename within that directory
        input_basename = os.path.basename(args.input)
        suffix = '_time_env.json' if args.time else '_env.json'
        output_filename = input_basename.replace('.xml', suffix)
        args.output = os.path.join(args.output, output_filename)
    
    print(f"Converting {args.input} to Gymnasium environment...")
    print(f"Output will be saved to: {args.output}")
    
    try:
        # Parse the XML file
        tree = parse_file(args.input)
        
        # Apply pruning if requested
        if args.prune:
            print(f"Pruning tree to keep subtree rooted at: {args.prune}")
            tree = tree.prune(args.prune)
        
        # Create environment converter
        env_converter = TreeToGymEnvironment(tree, args.time)
        
        # Save environment
        env_converter.save_environment(args.output)
    
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
    except Exception as e:
        print(f"Error during conversion: {e}")

"""
Time-based Environment Logic:

When time_based=True, the environment implements temporal mechanics where:

1. Actions are split into 'start_X' and 'end_X' pairs:
   - start_X: Initiates the action, pays the cost, sets progress=1, and starts the timer
   - end_X: Completes the action when timer reaches 0, applies effects, resets progress

2. Time state variables:
   - time_attacker: Timer for attacker actions (-1 means can act, >0 means waiting)
   - time_defender: Timer for defender actions (-1 means can act, >0 means waiting)

3. Wait actions:
   - wait_attacker/wait_defender: Decrements the respective timer when >0

4. Action execution flow:
   - Player starts an action (if time < 0 and preconditions met)
   - Time is set to action's duration, player switches
   - Other player acts while first player waits
   - When timer reaches 0, player can end the action to apply effects

This creates realistic temporal constraints where actions take time to complete
and players must wait during action execution.
"""
