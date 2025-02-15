# Import all constants from the cnnClassifier.constants module
# This might include file paths, model parameters, and other predefined values
from cnnClassifier.constants import *  

# Import the os module for interacting with the operating system (e.g., file handling, path manipulation)
import os  

# Import utility functions from cnnClassifier.utils.common
from cnnClassifier.utils.common import read_yaml, create_directories, save_json  
# `read_yaml`: Reads YAML files and converts them into Python dictionaries
# `create_directories`: Ensures that specified directories exist or creates them
# `save_json`: Saves Python objects as JSON files

# Import various configuration entity classes from cnnClassifier.entity.config_entity
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig  # Configuration for data ingestion (e.g., downloading and preparing datasets)
)




# Define a configuration manager class to handle configuration settings
class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,  # Default path for the main configuration file
        params_filepath=PARAMS_FILE_PATH  # Default path for the parameters file
    ):
        # Read and parse YAML configuration files
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Ensure that the artifact root directory exists
        create_directories([self.config.artifacts_root])


    # Method to retrieve data ingestion configuration
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion  # Extract the data ingestion section from the config

        # Ensure the root directory for data ingestion exists
        create_directories([config.root_dir])

        # Create and return a DataIngestionConfig object with necessary parameters
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config  # Return the configuration object