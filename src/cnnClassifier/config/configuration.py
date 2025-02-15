from cnnClassifier.constants import *  # Importing all constants defined in cnnClassifier.constants
import os  # Importing os module for interacting with the operating system
from cnnClassifier.utils.common import read_yaml, create_directories, save_json  # Utility functions for reading YAML, creating directories, and saving JSON
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig)  # Importing configuration entity classes

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,  # Default configuration file path
        params_filepath=PARAMS_FILE_PATH  # Default parameters file path
    ):
        
        self.config = read_yaml(config_filepath)  # Reading configuration file
        self.params = read_yaml(params_filepath)  # Reading parameters file

        create_directories([self.config.artifacts_root])  # Creating the root directory for artifacts

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion  # Extracting data ingestion configuration

        create_directories([config.root_dir])  # Creating root directory for data ingestion

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,  # Path to the root directory
            source_URL=config.source_URL,  # URL of the dataset source
            local_data_file=config.local_data_file,  # Local file path for downloaded dataset
            unzip_dir=config.unzip_dir  # Directory where the dataset will be extracted
        )

        return data_ingestion_config  # Returning the configured data ingestion object

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model  # Extracting base model preparation configuration
        
        create_directories([config.root_dir])  # Creating root directory for base model preparation

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),  # Path to the root directory
            base_model_path=Path(config.base_model_path),  # Path to the base model file
            updated_base_model_path=Path(config.updated_base_model_path),  # Path to save the updated model
            params_image_size=self.params.IMAGE_SIZE,  # Image size parameter
            params_learning_rate=self.params.LEARNING_RATE,  # Learning rate parameter
            params_include_top=self.params.INCLUDE_TOP,  # Include top layers parameter
            params_weights=self.params.WEIGHTS,  # Model weights parameter
            params_classes=self.params.CLASSES  # Number of output classes
        )

        return prepare_base_model_config  # Returning the configured base model preparation object
