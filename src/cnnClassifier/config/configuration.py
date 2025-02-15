# Import constants and required modules
from cnnClassifier.constants import *  # Import all predefined constants
import os  # OS module for file and directory operations
from cnnClassifier.utils.common import read_yaml, create_directories, save_json  # Utility functions for reading YAML, creating directories, and saving JSON
from cnnClassifier.entity.config_entity import (  # Import configuration entity classes
    DataIngestionConfig, 
    PrepareBaseModelConfig, 
    TrainingConfig, 
    EvaluationConfig
)

# Define the ConfigurationManager class, responsible for managing configurations
class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,  # Default path for config YAML file
        params_filepath=PARAMS_FILE_PATH   # Default path for parameters YAML file
    ):
        """
        Initializes the ConfigurationManager by reading the configuration
        and parameter files and creating necessary directories.
        """
        self.config = read_yaml(config_filepath)  # Read configuration file
        self.params = read_yaml(params_filepath)  # Read parameters file

        create_directories([self.config.artifacts_root])  # Create the root directory for artifacts

    # ---------------------- DATA INGESTION CONFIGURATION ---------------------- #
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves and returns the configuration for data ingestion.
        """
        config = self.config.data_ingestion  # Extract data ingestion config from the YAML file

        create_directories([config.root_dir])  # Ensure the root directory for data ingestion exists

        # Create an instance of DataIngestionConfig with the required parameters
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config  # Return the data ingestion configuration object

    # ---------------------- PREPARE BASE MODEL CONFIGURATION ---------------------- #
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        Retrieves and returns the configuration for preparing the base model.
        """
        config = self.config.prepare_base_model  # Extract base model preparation config
        
        create_directories([config.root_dir])  # Ensure the root directory for the base model exists

        # Create an instance of PrepareBaseModelConfig with the necessary parameters
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config  # Return the base model configuration object

    # ---------------------- TRAINING CONFIGURATION ---------------------- #
    def get_training_config(self) -> TrainingConfig:
        """
        Retrieves and returns the configuration for model training.
        """
        training = self.config.training  # Extract training configuration
        prepare_base_model = self.config.prepare_base_model  # Extract base model preparation configuration
        params = self.params  # Extract training parameters

        # Define the training data path
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "kidney-ct-scan-image")

        create_directories([Path(training.root_dir)])  # Ensure the root directory for training exists

        # Create an instance of TrainingConfig with necessary parameters
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config  # Return the training configuration object

    # ---------------------- EVALUATION CONFIGURATION ---------------------- #
    def get_evaluation_config(self) -> EvaluationConfig:
        """
        Retrieves and returns the configuration for model evaluation.
        """
        # Create an instance of EvaluationConfig with necessary parameters
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",  # Path to the trained model file
            training_data="artifacts/data_ingestion/kidney-ct-scan-image",  # Path to training data for evaluation
            mlflow_uri="https://dagshub.com/abhishekpatel16/Kidney-Disease-Classificaion-End-to-End-MLflow-DVC.mlflow",  # MLflow tracking URI
            all_params=self.params,  # All hyperparameters and settings
            params_image_size=self.params.IMAGE_SIZE,  # Image size used for training
            params_batch_size=self.params.BATCH_SIZE  # Batch size used for training
        )
        return eval_config  # Return the evaluation configuration object
