# PANACEA

PANACEA (Platform for Automated Network Attack Countermeasure Evaluation and Analysis) provides tools to convert attack-defense trees from ADTool into different formats for analysis and simulation:

1. **PRISM model generation** for formal verification with PRISM-games
2. **PettingZoo environment generation** for multi-agent reinforcement learning
3. **Multi-agent reinforcement learning** with Actor-Critic algorithms
4. **Interactive analysis** through Jupyter notebooks

## Features

- **Tree-to-PRISM conversion**: Generate formal models for game-theoretic analysis
- **Tree-to-Environment conversion**: Create PettingZoo-compatible multi-agent environments
- **Time-based models**: Support for temporal constraints and timing analysis with wait actions
- **Multi-agent RL**: Train attacker and defender agents using Actor-Critic methods in alternating environments
- **Visualization**: Interactive tree visualization and training progress monitoring
- **Experimental framework**: Built-in experiment management and data collection

## Requirements

The project was tested on Windows 10 and Ubuntu WSL2 with the following software versions:
- Python 3.12+
- ADTool 2.2.2
- PRISM-games 3.2.1
- PyTorch 2.0+

## Installation

### Basic Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Marini97/PANACEA.git
cd PANACEA
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Optional: Jupyter Notebook Support

For interactive development and training visualization:

```bash
pip install jupyter ipywidgets
jupyter notebook
```

### ADTool

To generate the XML file, you need to use ADTool. You can download the software from the following link: [Download ADTool](https://satoss.uni.lu/members/piotr/adtool/) or with the following command:

```bash
wget https://satoss.uni.lu/members/piotr/adtool/ADTool-2.2.2-full.jar
```

To run the software, you need to use the following command:
```bash
java -jar ADTool-2.2.2-full.jar 
```
When creating the tree, every node needs to have a unique name and a description with the following fields:
```
Type: Action/Attribute/Goal
Action: if Type is Action then the action name
Cost: if Type is Action then the cost of the action
Time: if Type is Action then the time of the action
Role: Attacker/Defender
```
Example:
```
Type: Action
Action: exfiltrateData
Cost: 50
Time: 2
Role: Attacker
```
Some xml files are provided in the `trees` folder.

Once you have created the Tree, you can export it to an XML file to use it with the script.

### PRISM-games

To run the PRISM model, you need to use PRISM-games. You can download the software from the following link: [Download PRISM-games](https://www.prismmodelchecker.org/games/download.php)
or with the following command:

```bash
wget https://www.prismmodelchecker.org/dl/prism-games-3.2.1-linux64-x86.tar.gz
```
Unzip the file inside this project, specifying the correct path, with the following command:
```bash
tar -xvf prism-games-3.2.1-linux64-x86.tar.gz -C PANACEA
```

Install the software with the following command:
```bash
cd PANACEA/prism-games-3.2.1-linux64-x86
./install.sh
```
To run PRISM-games: for Windows, double-click the short-cut; on other OSs, run `bin/xprism` for the GUI or `bin/prism` for the command-line version.


## Usage

### Generating PRISM Models

To generate a PRISM model from an ADTool XML file, use `main.py`:

```bash
python tree_to_prism.py --input trees/adt_nuovo.xml --output output.prism --time
```

Options:
- `--input` or `-i`: Path to the XML file from ADTool
- `--output` or `-o`: Path to the output PRISM model file
- `--props`: Generate properties for the PRISM model
- `--prune` or `-p`: Name of the subtree root to keep
- `--time` or `-t`: Generate a time-based PRISM model

### Generating PettingZoo Environments

To generate a JSON environment for multi-agent reinforcement learning use `tree_to_env.py`:

```bash
python tree_to_env.py --input trees/adt_nuovo.xml --output rl/envs/my_env.json
```

Options:
- `--input` or `-i`: Path to the XML file from ADTool (required)
- `--output` or `-o`: Path to the output JSON environment file
- `--prune` or `-p`: Name of the subtree root to keep

### Running PettingZoo Environments

To run the generated PettingZoo environment, you can use the provided example scripts:

```bash
# Standard multi-agent environment
python rl/adt_env_example.py --env rl/envs/my_env.json --mode random
python rl/adt_env_example.py --env rl/envs/my_env.json --mode interactive

# Time-based environment with action durations
python rl/adt_time_env_example.py --env rl/envs/my_env.json --mode random
python rl/adt_time_env_example.py --env rl/envs/my_env.json --mode interactive
```

The environment supports:
- **Alternating turns**: Agents take turns following PettingZoo's AEC (Agent Environment Cycle) pattern
- **Action preconditions**: Actions are only available when preconditions are met
- **Time mechanics**: Time-based environments include action durations and wait actions
- **Reward structure**: Cost-based rewards and terminal rewards for achieving goals

### Multi-Agent Reinforcement Learning

Train attacker and defender agents using Actor-Critic algorithms in PettingZoo environments:

```bash
# Using the training notebooks (recommended)
jupyter notebook rl/adt_training.ipynb
# or for time-based environments
jupyter notebook rl/adt_time_training.ipynb
```

The training process includes:
- **PettingZoo AEC environments**: Alternating Environment Cycle for turn-based gameplay
- **Multi-agent Actor-Critic**: Separate networks for attacker and defender agents
- **Policy gradient training**: Independent learning with shared environment state
- **Performance monitoring**: Real-time visualization of training progress and win rates
- **Model persistence**: Automatic saving of trained models and training logs

Available environments:
- **Standard ADT environment**: Basic turn-based attack-defense gameplay
- **Time-based ADT environment**: Includes action durations, wait mechanics, and temporal constraints

### Interactive Analysis

Use the demo notebook for tree visualization and PRISM model generation:

```bash
jupyter notebook demo_prism.ipynb
```

This notebook provides:
- Tree structure visualization
- Attack-defense tree analysis
- PRISM model generation and export
- Interactive exploration of tree properties

### Running PRISM Models

Then you can run the PRISM model with the following command:
```bash
cd PANACEA/prism-games-3.2.1-linux64-x86
bin/prism output.prism properties.props -prop 1 -exportresults output/results.csv:csv -exportstrat output/strat.dot
```

Or you can run PRISM-games with the GUI:
```bash
cd PANACEA/prism-games-3.2.1-linux64-x86
bin/xprism
```
And then load the PRISM model and the properties file.

## Project Structure

```
PANACEA/
├── tree.py                    # Core tree data structure
├── tree_to_prism.py          # PRISM model generation
├── tree_to_env.py            # PettingZoo environment generation
├── demo_prism.ipynb          # Interactive demo notebook
├── requirements.txt          # Python dependencies
├── trees/                    # Sample ADT XML files
├── rl/                       # Reinforcement learning components
│   ├── adt_env.py           # Standard PettingZoo multi-agent environment
│   ├── adt_time_env.py      # Time-based PettingZoo environment
│   ├── adt_actor_critic.py  # Actor-Critic implementation
│   ├── adt_env_example.py   # Example usage for standard environment
│   ├── adt_time_env_example.py  # Example usage for time environment
│   ├── adt_training.ipynb   # Training notebook for standard environment
│   ├── adt_time_training.ipynb  # Training notebook for time environment
│   ├── envs/                # JSON environment specifications
│   ├── trained_models/      # Saved model checkpoints
│   └── training_logs/       # Training progress logs
└── experiments/             # Experimental framework
    ├── run_experiment*.sh   # Experiment scripts
    └── utils/              # Experiment utilities
```

## Advanced Usage

### Custom Tree Creation

When creating trees in ADTool, ensure each node has the required metadata in the description field:

```
Type: Action/Attribute/Goal
Action: [action_name]
Cost: [numeric_cost]
Time: [numeric_time]
Role: Attacker/Defender
```

### Model Training Configuration

The Actor-Critic training can be customized through various parameters:
- **Learning rates**: Separate rates for actor and critic networks
- **Network architecture**: Hidden layer sizes, activation functions
- **Training parameters**: Episodes, batch sizes, update frequencies
- **Reward shaping**: Cost penalties, terminal rewards, win/loss bonuses
- **Environment settings**: Action availability, precondition logic, time constraints

### PettingZoo Environment Structure

The generated JSON environments follow this structure:
- **State space**: Discrete variables for goals, attributes, and agent turns
- **Action space**: Discrete actions with preconditions and effects
- **Transitions**: State changes and agent turn switching
- **Rewards**: Action costs, terminal rewards, and penalties
- **Time mechanics**: Action durations and wait states (time-based environments only)

### Experimental Framework

Run systematic experiments using the provided scripts:

```bash
# Run experiment with memory configuration
cd experiments
./run_experiment1.sh 4g 1g  # JVM memory, CUDD memory
```

## Contributing

When contributing to this project:
1. Ensure all dependencies are properly documented in requirements.txt
2. Update both README and requirements.txt for new features 
3. Include appropriate tests and documentation
4. Follow the existing code structure and naming conventions
5. Test both standard and time-based PettingZoo environments
6. Verify compatibility with the Actor-Critic training pipeline