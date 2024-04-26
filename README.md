# IRS

The project is a simple script that reads an XML file from ADTool and generates a PRISM model for PRISM-games.

The project was tested on Windows 10 with the following software versions:
- Python 3.10
- ADTool 2.2.2
- PRISM-games 3.2.1

## Installation

To install the project, you need to clone the repository and install the dependencies.

```bash
git clone https://github.com/Marini97/IRS.git
cd IRS
pip install -r requirements.txt
```

## Usage

To use the project, you need to run the script `main.py` with the following arguments:
- `--input` or `-i`: the path to the XML file from ADTool
- `--output` or `-o`: the path to the output file for the PRISM model

```bash
python main.py --input input.xml --output output.prism
```

