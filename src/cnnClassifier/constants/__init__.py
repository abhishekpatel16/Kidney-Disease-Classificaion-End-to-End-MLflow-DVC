# Import the Path class from the pathlib module
# Path provides an object-oriented way to handle file system paths
from pathlib import Path  

# Define the path to the main configuration file
CONFIG_FILE_PATH = Path("config/config.yaml")  
# This represents the file path "config/config.yaml" in a cross-platform way

# Define the path to the parameters file
PARAMS_FILE_PATH = Path("params.yaml")  
# This represents the file path "params.yaml", typically used for storing hyperparameters or settings
