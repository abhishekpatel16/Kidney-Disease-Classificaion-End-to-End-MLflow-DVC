from cnnClassifier.constants import *  # Importing constants used in the configuration
import os  # Importing OS module for file path operations
from pathlib import Path  # Importing Path for handling file paths
from cnnClassifier.utils.common import read_yaml, create_directories, save_json  # Importing utility functions for reading YAML, creating directories, and saving JSON files
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig)  # Importing data classes for configuration management


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,  # Default configuration file path
        params_filepath=PARAMS_FILE_PATH  # Default parameters file path
    ):
        """
        Initializes the ConfigurationManager by reading YAML configuration files and creating necessary directories.
        """
        
        self.config = read_yaml(config_filepath)  # Reading the main configuration file into a dictionary
        self.params = read_yaml(params_filepath)  # Reading the parameters file into a dictionary

        create_directories([self.config.artifacts_root])  # Creating the root directory where artifacts will be stored

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Method to get data ingestion configuration.
        Reads the ingestion settings from the config file and ensures required directories are created.
        Returns:
            DataIngestionConfig: Configuration object for data ingestion.
        """
        config = self.config.data_ingestion  # Extracting the data ingestion configuration settings

        create_directories([config.root_dir])  # Creating the root directory for data ingestion

        # Creating and returning a DataIngestionConfig object with extracted parameters
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,  # Root directory for storing data
            source_URL=config.source_URL,  # URL to download the dataset from
            local_data_file=config.local_data_file,  # Path where the dataset will be stored locally
            unzip_dir=config.unzip_dir  # Directory where the dataset will be extracted
        )
        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        Method to get base model preparation configuration.
        Reads settings from the config file, ensures required directories exist, and returns the configuration object.
        Returns:
            PrepareBaseModelConfig: Configuration object for preparing the base model.
        """
        config = self.config.prepare_base_model  # Extracting base model preparation config settings
        
        create_directories([config.root_dir])  # Creating the directory where the base model will be stored

        # Creating and returning a PrepareBaseModelConfig object with necessary configurations
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),  # Root directory for base model
            base_model_path=Path(config.base_model_path),  # Path where the pre-trained base model is stored
            updated_base_model_path=Path(config.updated_base_model_path),  # Path for saving the updated base model
            params_image_size=self.params.IMAGE_SIZE,  # Image size parameter for the model
            params_learning_rate=self.params.LEARNING_RATE,  # Learning rate for training
            params_include_top=self.params.INCLUDE_TOP,  # Whether to include the top layer of the model
            params_weights=self.params.WEIGHTS,  # Pre-trained weights to use
            params_classes=self.params.CLASSES  # Number of output classes for the model
        )
        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:
        """
        Method to get training configuration.
        Extracts training settings, ensures necessary directories exist, and returns a configuration object.
        Returns:
            TrainingConfig: Configuration object for model training.
        """
        training = self.config.training  # Extracting training-related configuration
        prepare_base_model = self.config.prepare_base_model  # Extracting base model preparation settings
        params = self.params  # Extracting model hyperparameters

        # Constructing the path to the training dataset
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "kidney-ct-scan-image")
        
        create_directories([Path(training.root_dir)])  # Creating the directory where training-related artifacts will be stored

        # Creating and returning a TrainingConfig object with extracted parameters
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),  # Directory for storing training results
            trained_model_path=Path(training.trained_model_path),  # Path where the trained model will be saved
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),  # Path to the updated base model
            training_data=Path(training_data),  # Path to the training dataset
            params_epochs=params.EPOCHS,  # Number of training epochs
            params_batch_size=params.BATCH_SIZE,  # Batch size for training
            params_is_augmentation=params.AUGMENTATION,  # Whether to apply data augmentation
            params_image_size=params.IMAGE_SIZE  # Image size parameter
        )
        return training_config
