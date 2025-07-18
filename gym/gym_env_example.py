"""
Example implementation demonstrating how to use the Attack-Defense Tree 
Gymnasium environment from adt_env.py

This demonstrates how the saved JSON can be used to instantiate
a working Gymnasium environment for reinforcement learning.
"""

import numpy as np
from adt_env import AttackDefenseTreeEnv


def demo_environment():
    """Demonstrate the environment with random actions."""
    print("=== Attack-Defense Tree Environment Demo ===")
    
    # Load environment
    env = AttackDefenseTreeEnv("gym/envs/adt_nuovo_env.json", render_mode="human")

    print(f"Environment loaded successfully!")
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Goal: {env.goal}")
    
    # Reset environment
    obs, info = env.reset()
    env.render()
    
    # Run a few steps
    step = 0
    max_steps = 10
    
    while step < max_steps:
        print(f"\n--- Step {step + 1} ---")
        
        # Get available actions
        available_actions = env.get_available_actions()
        print(f"Available actions: {available_actions}")
        
        if available_actions:
            # Choose random available action
            action = np.random.choice(available_actions)
        else:
            # Choose wait action (last action in list)
            current_player = "attacker" if env.state["current_player"] == 0 else "defender"
            if current_player == "attacker":
                action = env.num_attacker_actions - 1
            else:
                action = env.num_defender_actions - 1
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Action taken: {info['action_taken']}")
        print(f"Valid action: {info['action_valid']}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        
        env.render()
        
        if terminated or truncated:
            print(f"\nEpisode ended after {step + 1} steps!")
            break
        
        step += 1
    
    env.close()


def interactive_environment():
    """Interactive mode where user can choose actions at each step."""
    print("=== Interactive Attack-Defense Tree Environment ===")
    
    # Load environment
    env = AttackDefenseTreeEnv("gym/envs/adt_nuovo_env.json", render_mode="human")

    print(f"Environment loaded successfully!")
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Goal: {env.goal}")
    print("\nInstructions:")
    print("- You will be prompted to choose actions for both attacker and defender")
    print("- Enter the action number or press Enter for random action")
    print("- Type 'quit' to exit early")
    
    # Reset environment
    obs, info = env.reset()
    env.render()
    
    step = 0
    
    while True:
        step += 1
        print(f"\n{'='*50}")
        print(f"Step {step}")
        print(f"{'='*50}")
        
        current_player = "attacker" if env.state["current_player"] == 0 else "defender"
        print(f"Current Player: {current_player.upper()}")
        
        # Get available actions
        available_actions = env.get_available_actions()
        
        # Display action options
        if current_player == "attacker":
            actions = env.attacker_actions
        else:
            actions = env.defender_actions
        
        print(f"\nAvailable actions for {current_player}:")
        if not available_actions:
            print("  No valid actions available - will use wait action")
            action = len(actions) - 1
        else:
            for action_id in available_actions:
                action_spec = actions[str(action_id)]
                print(f"  [{action_id}] {action_spec['name']} (cost: {action_spec['cost']})")
            
            # Get user input
            try:
                user_input = input(f"\nChoose action for {current_player} (0-{len(available_actions)-1}, Enter for random, 'quit' to exit): ").strip()
                
                if user_input.lower() == 'quit':
                    print("Exiting interactive mode...")
                    break
                elif user_input == "":
                    # Random action from available
                    action = np.random.choice(available_actions)
                    print(f"Randomly selected action: {action}")
                else:
                    action = int(user_input)
                    if action not in available_actions:
                        print(f"Invalid action {action}. Using random available action.")
                        action = np.random.choice(available_actions)
            except ValueError:
                print("Invalid input. Using random available action.")
                action = np.random.choice(available_actions)
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nAction executed: {info['action_taken']}")
        print(f"Valid action: {info['action_valid']}")
        print(f"Reward for {current_player}: {reward}")
        print(f"Terminated: {terminated}")
        
        env.render()
        
        if terminated or truncated:
            print(f"\n{'='*50}")
            print(f"GAME OVER! Episode ended after {step} steps!")
            if env.state[env.goal] == 1:
                print("🎯 ATTACKER WON! Goal achieved!")
            else:
                print("🛡️  DEFENDER WON! Attack prevented!")
            print(f"{'='*50}")
            break
    
    env.close()


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
    mode = get_mode_choice()
    if mode == "random":
        demo_environment()
    elif mode == "interactive":
        interactive_environment()
    else:
        print("No mode selected. Exiting...")
