import numpy as np
import random
import argparse
import os
from adt_multiagent_env import AttackDefenseTreeMultiAgentEnv 

def demo_environment(env_file):
    """Demonstrate the environment with random actions."""
    print("=== Attack-Defense Tree Multi-Agent Environment Demo ===")
    print(f"Loading environment from: {env_file}")
    
    # Load AEC (turn-based) environment
    env = AttackDefenseTreeMultiAgentEnv(env_file, render_mode="human")

    print(f"Environment loaded successfully!")
    print(f"Possible agents: {env.possible_agents}")
    print(f"Action spaces: {env.action_spaces}")
    print(f"Observation spaces: {[space.shape for space in env.observation_spaces.values()]}")
    
    # Reset environment
    env.reset(seed=42)
    
    # Run a few steps
    step = 0
    max_steps = 10
    
    for agent in env.agent_iter():
        if step >= max_steps:
            break
            
        observation, reward, termination, truncation, info = env.last()
        
        print(f"\n--- Step {step + 1}: Agent {agent} ---")
        print(f"Reward: {reward}")
        print(f"Terminated: {termination}, Truncated: {truncation}")
        
        if termination or truncation:
            action = 0  # Use default action when terminated
        else:
            # Get available actions
            available_actions = env.get_available_actions(agent)
            print(f"Available actions: {available_actions}")
            
            if available_actions:
                # Choose random available action
                action = np.random.choice(available_actions)
                print(f"Chosen action: {action}")
            else:
                # No valid actions available - game will terminate
                action = 0
                print(f"🚫 No valid actions available - {agent} is blocked!")
        
        env.step(action)
        env.render()
        
        step += 1
        
        # Check if all agents are terminated
        if all(env.terminations.values()):
            print(f"\n🎮 GAME OVER after {step} steps!")
            print(f"Final rewards: {env.rewards}")
            break
    
    env.close()


def interactive_environment(env_file):
    """Interactive mode where user can choose actions at each step."""
    print("=== Interactive Attack-Defense Tree Multi-Agent Environment ===")
    print(f"Loading environment from: {env_file}")
    
    # Load AEC (turn-based) environment
    env = AttackDefenseTreeMultiAgentEnv(env_file, render_mode="human")

    print(f"Environment loaded successfully!")
    print(f"Possible agents: {env.possible_agents}")
    print(f"Action spaces: {env.action_spaces}")
    print("\nInstructions:")
    print("- You will be prompted to choose actions for each agent")
    print("- Enter the action number or press Enter for random action")
    print("- Type 'quit' to exit early")
    
    # Reset environment
    env.reset(seed=42)
    env.render()
    
    step = 0
    
    for agent in env.agent_iter():
        step += 1
        print(f"\n{'='*50}")
        print(f"Step {step}: Agent {agent.upper()}")
        print(f"{'='*50}")
        
        observation, reward, termination, truncation, info = env.last()
        
        print(f"Previous reward: {reward}")
        
        if termination or truncation:
            print(f"Agent {agent} is terminated/truncated")
            env.step(0)  # Use default action when terminated
            continue
        
        # Get available actions
        available_actions = env.get_available_actions(agent)
        
        if not available_actions:
            print(f"🚫 No valid actions available for {agent}!")
            action = 0  # Will be handled as invalid
        else:
            print(f"Available actions for {agent}:")
            
            # Get action details for display
            if agent == "attacker":
                actions_dict = env.attacker_actions
            else:
                actions_dict = env.defender_actions
            
            for action_id in available_actions:
                action_spec = actions_dict[str(action_id)]
                print(f"  [{action_id}] {action_spec['name']} (cost: {action_spec.get('cost', 0)})")
            
            # Get user input
            try:
                user_input = input(f"\nChoose action for {agent} (Enter for random, 'quit' to exit): ").strip()
                
                if user_input.lower() == 'quit':
                    print("Exiting...")
                    break
                elif user_input == "":
                    # Random choice
                    action = random.choice(available_actions)
                    print(f"Randomly chose action: {action}")
                else:
                    action = int(user_input)
                    if action not in available_actions:
                        print(f"Invalid action {action}. Choosing random action.")
                        action = random.choice(available_actions)
            except ValueError:
                print("Invalid input. Choosing random action.")
                action = random.choice(available_actions)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
        
        print(f"Taking action: {action}")
        env.step(action)
        env.render()
        
        # Check if game is over
        if all(env.terminations.values()):
            print(f"\n🎮 GAME OVER after {step} steps!")
            print(f"Final rewards: {env.rewards}")
            
            # Determine winner
            if env.rewards["attacker"] > env.rewards["defender"]:
                print("🎯 ATTACKER WINS!")
            elif env.rewards["defender"] > env.rewards["attacker"]:
                print("🛡️  DEFENDER WINS!")
            else:
                print("🤝 It's a tie!")
            break
    
    env.close()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Attack-Defense Tree Multi-Agent Environment Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use default environment
  %(prog)s -e envs/my_env.json               # Use specific environment file
  %(prog)s -e envs/my_env.json -m interactive # Use interactive mode
        """
    )
    
    parser.add_argument(
        '-e', '--env-file',
        type=str,
        default='envs/adt_nuovo_env.json',
        help='Path to the environment JSON file (default: envs/adt_nuovo_env.json)'
    )
    
    parser.add_argument(
        '-m', '--mode',
        choices=['random', 'interactive'],
        help='Demo mode: random actions or interactive mode'
    )
    
    return parser.parse_args()


def validate_env_file(env_file):
    """Validate that the environment file exists and is readable."""
    if not os.path.exists(env_file):
        raise FileNotFoundError(f"Environment file not found: {env_file}")
    
    if not os.path.isfile(env_file):
        raise ValueError(f"Path is not a file: {env_file}")
    
    if not env_file.endswith('.json'):
        print(f"Warning: File doesn't have .json extension: {env_file}")
    
    return env_file


def get_mode_choice():
    """Get user's choice for demo mode."""
    print("Choose demo mode:")
    print("1. Random actions (original demo)")
    print("2. Interactive mode (choose actions)")
    
    while True:
        try:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == "1":
                return "random"
            elif choice == "2":
                return "interactive"
            else:
                print("Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


if __name__ == "__main__":
    args = parse_arguments()
    
    try:
        # Validate environment file
        env_file = validate_env_file(args.env_file)
        print(f"Using environment file: {env_file}")
        
        # Determine mode
        if args.mode:
            mode = args.mode
        else:
            mode = get_mode_choice()
        
        # Run demo
        if mode == "random":
            demo_environment(env_file)
        elif mode == "interactive":
            interactive_environment(env_file)
        else:
            print("No mode selected. Exiting...")
            
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        exit(1)
    except KeyboardInterrupt:
        print("\nExiting...")
        exit(0)
