# PANACEA

This project provides tools to convert attack-defense trees from ADTool into different formats for analysis and simulation:
1. PRISM model generation for formal verification with PRISM-games
2. Gymnasium environment generation for reinforcement learning


The project was tested on Windows 10 and Ubuntu WSL2 with the following software versions:
- Python 3.12
- ADTool 2.2.2
- PRISM-games 3.2.1

## Installation

To install the project, you need to clone the repository and install the dependencies.

```bash
git clone https://github.com/Marini97/PANACEA.git
cd PANACEA
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

### Generating Gymnasium Environments

To generate a JSON environment use `tree_to_env.py`:

```bash
python tree_to_env.py --input trees/adt_nuovo.xml --output gym/envs/my_env.json
python tree_to_env.py --input trees/adt_nuovo.xml --output gym/envs/my_time_env.json --time
```

Options:
- `--input` or `-i`: Path to the XML file from ADTool (required)
- `--output` or `-o`: Path to the output JSON environment file
- `--time` or `-t`: Generate time-based environment with temporal constraints
- `--prune` or `-p`: Name of the subtree root to keep
- `--verbose` or `-v`: Enable verbose output

### Running Gymnasium Environments
To run the generated Gymnasium environment, you can use the following Python script:

```bash
python gym/gym_env_example.py --env gym/envs/my_env.json
```

### Running PRISM Models

Then you can run the PRISM model with the following command:
```bash
cd PANACEA/prism-games-3.2.1-linux64-x86
/bin/prism output.prism properties.props -prop 1 -exportresults output/results.csv:csv  -exportstrat output/strat.dot
```

Or you can run PRISM-games with the GUI with the following command:
```bash
cd PANACEA/prism-games-3.2.1-linux64-x86
/bin/xprism
```
And then load the PRISM model and the properties file.