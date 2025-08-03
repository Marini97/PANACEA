import json
import os
import argparse
from typing import Dict, List, Tuple, Set, Any, Optional
from tree import Tree
from tree_to_prism import parse_file, get_info


def get_environment(tree):
    """
    Converts a tree object into a Gymnasium environment specification.
    
    Args:
        tree: The tree object to be converted.
        
    Returns:
        dict: A dictionary containing the complete Gymnasium environment specification.
    """
    df = tree.to_dataframe()
    goal, actions_to_goal, initial_attributes, attacker_actions, defender_actions, df_attacker, df_defender = get_info(df)
    
    # Initialize environment specification
    env_spec = {
        "metadata": {
            "goal": goal,
            "actions_to_goal": list(actions_to_goal),
            "initial_attributes": initial_attributes
        },
        "state_space": {},
        "action_space": {},
        "transitions": {},
        "rewards": {},
        "initial_state": {},
        "terminal_states": []
    }
    
    # Build environment components
    _define_state_space(env_spec, goal, df_attacker, df_defender, initial_attributes)
    _define_action_space(env_spec, attacker_actions, defender_actions)
    _define_initial_state(env_spec, goal, initial_attributes)
    _define_transitions(env_spec, attacker_actions, defender_actions)
    _define_rewards(env_spec, attacker_actions, defender_actions, actions_to_goal)
    _define_terminal_states(env_spec, goal)
    
    return env_spec


# Helper functions for building environment components

def _define_state_space(env_spec, goal, df_attacker, df_defender, initial_attributes):
    """Define the state space for the environment."""
    # Goal state
    env_spec["state_space"][goal] = {
        "type": "discrete",
        "low": 0,
        "high": 1,
    }
    
    # Current player turn
    env_spec["state_space"]["current_player"] = {
        "type": "discrete", 
        "low": 0,
        "high": 1,
    }
    
    # Attacker attributes (can be 0=not achieved, 1=achieved, 2=defended)
    for attr in set(df_attacker.loc[df_attacker["Type"] == "Attribute"]["Label"].values):
        env_spec["state_space"][attr] = {
            "type": "discrete",
            "low": 0,
            "high": 2,
        }
    
    # Initial system attributes (start as 1=vulnerable, can be 2=protected)
    for attr in initial_attributes:
        env_spec["state_space"][attr] = {
            "type": "discrete",
            "low": 1,
            "high": 2,
        }
    
    # Defender attributes
    defender_attributes = set(df_defender.loc[df_defender["Type"] == "Attribute"]["Label"].values)
    for attr in defender_attributes:
        env_spec["state_space"][attr] = {
            "type": "discrete",
            "low": 0,
            "high": 1,
        }


def _define_action_space(env_spec, attacker_actions, defender_actions):
    """Define available actions for both players."""
    env_spec["action_space"]["attacker"] = {
        "type": "discrete",
        "actions": {}
    }
    
    env_spec["action_space"]["defender"] = {
        "type": "discrete", 
        "actions": {}
    }
    
    _define_standard_action_space(env_spec, attacker_actions, defender_actions)


def _define_standard_action_space(env_spec, attacker_actions, defender_actions):
    """Define action space for non-time-based environment."""
    # Add attacker actions
    action_id = 0
    for action_name, action_data in attacker_actions.items():
        cost = int(action_data["cost"]) if action_data["cost"] else 0
        time = int(action_data["time"]) if action_data["time"] else 0
        
        # Build conditions list - only include the original preconditions
        conditions = action_data["preconditions"].copy()
        
        # Determine logic based on refinement type
        logic = "OR" if action_data["refinement"] == "disjunctive" else "AND"
        
        env_spec["action_space"]["attacker"]["actions"][action_id] = {
            "name": action_name,
            "preconditions": {
                "conditions": conditions,
                "logic": logic
            },
            "effect": action_data["effect"],
            "cost": cost,
            "time": time,
            "refinement": action_data["refinement"]
        }
        action_id += 1
    
    # Add defender actions
    action_id = 0
    for action_name, action_data in defender_actions.items():
        cost = int(action_data["cost"]) if action_data["cost"] else 0
        time = int(action_data["time"]) if action_data["time"] else 0
        
        # Build conditions list - only include the original preconditions
        conditions = action_data["preconditions"].copy()
        
        # Determine logic based on refinement type
        logic = "OR" if action_data["refinement"] == "disjunctive" else "AND"
        
        env_spec["action_space"]["defender"]["actions"][action_id] = {
            "name": action_name,
            "preconditions": {
                "conditions": conditions,
                "logic": logic
            },
            "effect": action_data["effect"],
            "cost": cost,
            "time": time,
            "refinement": action_data["refinement"]
        }
        action_id += 1


def _define_initial_state(env_spec, goal, initial_attributes):
    """Define the initial state of the environment."""
    env_spec["initial_state"] = {
        goal: 0,  # Goal not achieved initially
        "current_player": 0  # Attacker starts first
    }
    
    # Set initial attributes to vulnerable (1)
    for attr in initial_attributes:
        env_spec["initial_state"][attr] = 1


def _define_transitions(env_spec, attacker_actions, defender_actions):
    """Define state transitions for the environment."""
    env_spec["transitions"] = {
        "attacker": {},
        "defender": {}
    }
    
    # Define transitions for each action
    # Attacker actions
    for action_name, action_data in attacker_actions.items():
        env_spec["transitions"]["attacker"][action_name] = {
            "effects": {
                action_data["effect"]: 1,  # Achieve the effect
                "current_player": 1  # Switch to defender
            }
        }
        
        # Defender actions  
        for action_name, action_data in defender_actions.items():
            effect = action_data["effect"]
            # Defender actions should set effects to 2 to indicate protection/deactivation
            # This applies to both initial system attributes and attacker-generated attributes
            env_spec["transitions"]["defender"][action_name] = {
                "effects": {
                    effect: 2,  # Always set to 2 for defender protection
                    "current_player": 0  # Switch to attacker
                }
            }


def _define_rewards(env_spec, attacker_actions, defender_actions, actions_to_goal):
    """Define reward structure."""
    env_spec["rewards"] = {
        "attacker": {},
        "defender": {},
        "cost_penalty": 10,   # General cost penalty multiplier
        "terminal_rewards": {}  # Terminal state rewards
    }
    
    # Action costs as negative rewards
    for action_name, action_data in attacker_actions.items():
        cost = int(action_data["cost"]) if action_data["cost"] else 0
        env_spec["rewards"]["attacker"][action_name] = -cost
    
    for action_name, action_data in defender_actions.items():
        cost = int(action_data["cost"]) if action_data["cost"] else 0
        env_spec["rewards"]["defender"][action_name] = -cost
    
    # Terminal state penalties for actions that lead to goal
    for action in actions_to_goal:
        if action in attacker_actions:
            cost = int(attacker_actions[action]["cost"]) if attacker_actions[action]["cost"] else 0
            env_spec["rewards"]["terminal_rewards"][action] = -cost * env_spec["rewards"]["cost_penalty"]


def _define_terminal_states(env_spec, goal):
    """Define terminal conditions."""
    env_spec["terminal_states"] = [
        {
            "condition": f"{goal} == 1",
        }
    ]


def save_environment(env_spec, file):
    """Save the environment specification to a JSON file."""
    with open(file, 'w') as f:
        json.dump(env_spec, f, indent=2)


def convert_tree_to_gym_env(xml_file: str, output_file: Optional[str] = None):
    """
    Convert an XML tree file to a Gymnasium environment specification.
    
    Args:
        xml_file (str): Path to the XML tree file
        output_file (str): Path to save the JSON environment specification
                          If None, uses the XML filename with .json extension
    
    Returns:
        Dict: The environment specification
    """
    # Parse the XML file
    tree = parse_file(xml_file)
    
    # Generate environment specification
    env_spec = get_environment(tree)
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = xml_file.replace('.xml', '_env.json')
    
    # Save environment
    save_environment(env_spec, output_file)
    print(f"Environment saved to {output_file}")
    
    return env_spec


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
    args = parser.parse_args()
    
    # Generate output filename if not provided or if directory is provided
    if args.output is None:
        args.output = args.input.replace('.xml', '_env.json')
    elif os.path.isdir(args.output):
        # If output is a directory, generate filename within that directory
        input_basename = os.path.basename(args.input)
        output_filename = input_basename.replace('.xml', '_env.json')
        args.output = os.path.join(args.output, output_filename)
    
    print(f"Converting {args.input} to Gymnasium environment...")
    print(f"Output will be saved to: {args.output}")
    
    try:
        # Parse the XML file and apply pruning if requested
        tree = parse_file(args.input)
        if args.prune:
            print(f"Pruning tree to keep subtree rooted at: {args.prune}")
            tree = tree.prune(args.prune)
        
        # Generate environment specification
        env_spec = get_environment(tree)
        
        # Save environment
        save_environment(env_spec, args.output)
        print(f"Environment saved to {args.output}")
    
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
    except Exception as e:
        print(f"Error during conversion: {e}")
