import os  # Importing the os module for interacting with the operating system
from pathlib import Path  # Importing Path from pathlib to handle file paths more effectively
import logging  # Importing logging module for logging messages

# Setting up logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Defining the project name
project_name = 'cnnClassifier'

# List of files and directories to be created
list_of_files = [
    ".github/workflows/.gitkeep",  # GitHub workflows directory with a placeholder file
    f"src/{project_name}/__init__.py",  # Init file for main project module
    f"src/{project_name}/components/__init__.py",  # Init file for components module
    f"src/{project_name}/utils/__init__.py",  # Init file for utils module
    f"src/{project_name}/config/__init__.py",  # Init file for config module
    f"src/{project_name}/config/configuration.py",  # Configuration script
    f"src/{project_name}/pipeline/__init__.py",  # Init file for pipeline module
    f"src/{project_name}/entity/__init__.py",  # Init file for entity module
    f"src/{project_name}/constants/__init__.py",  # Init file for constants module
    "config/config.yaml",  # Configuration file in YAML format
    "dvc.yaml",  # DVC (Data Version Control) pipeline file
    "params.yaml",  # Parameters file
    "requirements.txt",  # Dependencies list
    "setup.py",  # Setup script for packaging
    "research/trials.ipynb",  # Jupyter notebook for research
    "templates/index.html"  # HTML template file
]

# Iterating through the list of files to create necessary directories and files
for filepath in list_of_files:
    filepath = Path(filepath)  # Converting string path to Path object for better handling
    filedir, filename = os.path.split(filepath)  # Splitting directory and filename

    # If the directory does not exist, create it
    if filedir:
        os.makedirs(filedir, exist_ok=True)  # Creating directory if it does not exist
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # Creating an empty file if it does not exist or has zero size
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass  # Creating an empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
