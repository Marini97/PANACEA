import argparse

from core import ADTTrainer
from core.adt_env import AttackDefenseTreeMultiAgentEnv
from core.adt_time_env import AttackDefenseTreeMultiAgentTimeEnv


def parse_config():
    """Create configuration with environment type selection."""
    parser = argparse.ArgumentParser(description='ADT Environment Trainer')
    
    # Environment type selection
    parser.add_argument('--env-type', type=str, choices=['basic', 'time'], default='time',
                        help='Type of ADT environment to train (basic or time)')
    
    # Environment settings
    parser.add_argument('--env-file', type=str, default='envs/adt_nuovo_env.json',
                        help='Path to environment configuration file')
    
    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Size of hidden layers in neural networks')
    parser.add_argument('--max-episodes', type=int, default=5000,
                        help='Maximum number of training episodes')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Maximum steps per episode')
    
    # Epsilon exploration parameters
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Starting epsilon value for exploration')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                        help='Final epsilon value for exploration')
    parser.add_argument('--epsilon-decay', type=int, default=5000,
                        help='Number of steps for epsilon decay')
    
    # Training control
    parser.add_argument('--update-frequency', type=int, default=10,
                        help='Update policy every N episodes')
    parser.add_argument('--print-frequency', type=int, default=100,
                        help='Print statistics every N episodes')
    parser.add_argument('--save-frequency', type=int, default=1000,
                        help='Save intermediate models every N episodes')
    
    # System settings
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Directory settings
    parser.add_argument('--model-dir', type=str, default='results/trained_models',
                        help='Directory to save trained models')
    parser.add_argument('--log-dir', type=str, default='results/training_logs',
                        help='Directory to save training logs')
    
    # Testing
    parser.add_argument('--test-games', type=int, default=10,
                        help='Number of test games to run after training')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and only test existing models')
    parser.add_argument('--no-save-models', action='store_true',
                        help='Skip saving intermediate models during training')
    
    args = parser.parse_args()
    config = vars(args)
    
    return config


class UnifiedADTTrainer(ADTTrainer):
    """
    Trainer that can work with both basic and time-based ADT environments.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.env_type = config['env_type']
    
    def get_environment_class(self):
        """Get the appropriate environment class based on environment type."""
        if self.env_type == 'time':
            return AttackDefenseTreeMultiAgentTimeEnv
        elif self.env_type == 'basic':
            return AttackDefenseTreeMultiAgentEnv
        else:
            raise ValueError(f"Unknown environment type: {self.env_type}")
    
    def adjust_config_for_env_type(self):
        """Adjust configuration based on environment type."""
        # Add environment-specific prefixes to save directories using organized structure
        if self.env_type == 'time':
            self.config['model_dir'] = f"results/trained_models_time"
            self.config['log_dir'] = f"results/training_logs_time"
        else:
            self.config['model_dir'] = f"results/trained_models_basic"
            self.config['log_dir'] = f"results/training_logs_basic"


def main():
    """Main function for ADT environment training."""
    # Parse command line arguments
    config = parse_config()
    
    print(f"🚀 ADT {config['env_type'].title()} Environment Training with Epsilon Exploration")
    print("="*70)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    print("="*70)
    
    # Create trainer
    trainer = UnifiedADTTrainer(config)
    trainer.adjust_config_for_env_type()
    
    # Get the appropriate environment class
    env_class = trainer.get_environment_class()
    
    if not config['skip_training']:
        # Train the agents
        env, attacker_agent, defender_agent, metrics = trainer.train(
            env_class, 
            config['env_file']
        )
        
        # Test the trained models
        test_results = trainer.test_models(config['test_games'])
    else:
        print("⏭️ Skipping training, loading existing models for testing...")
        # Setup environment and agents for testing
        trainer.setup_environment(env_class, config['env_file'])
        trainer.setup_agents()
        
        try:
            trainer.load_models("final")
            test_results = trainer.test_models(config['test_games'])
        except FileNotFoundError:
            print("❌ No trained models found. Please run training first.")
            return
    
    print(f"\n✅ Training and testing completed successfully!")


if __name__ == "__main__":
    main()
