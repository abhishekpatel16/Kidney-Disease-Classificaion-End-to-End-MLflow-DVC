# Importing necessary modules  
from dataclasses import dataclass  # Enables defining immutable and structured data classes  
from pathlib import Path  # Provides a convenient way to handle file paths  

# Defining a dataclass to store configuration for data ingestion  
@dataclass(frozen=True)  # Makes the class immutable (frozen=True)
class DataIngestionConfig:
    root_dir: Path  # Directory where all ingestion-related files will be stored  
    source_URL: str  # URL to fetch the data from  
    local_data_file: Path  # Path where the downloaded file will be stored locally  
    unzip_dir: Path  # Directory where the downloaded file will be extracted  

# Defining a dataclass to store configuration for preparing the base model  
@dataclass(frozen=True)  
class PrepareBaseModelConfig:
    root_dir: Path  # Directory for storing base model-related files  
    base_model_path: Path  # Path to the pre-trained model  
    updated_base_model_path: Path  # Path to save the updated version of the base model  
    params_image_size: list  # List specifying image dimensions (e.g., [224, 224])  
    params_learning_rate: float  # Learning rate for fine-tuning the model  
    params_include_top: bool  # Whether to include the top layers in the base model  
    params_weights: str  # Specifies which weights to use (e.g., 'imagenet')  
    params_classes: int  # Number of classes in the dataset  

# Defining a dataclass to store configuration for training the model  
@dataclass(frozen=True)  
class TrainingConfig:
    root_dir: Path  # Directory for storing training-related files  
    trained_model_path: Path  # Path where the trained model will be saved  
    updated_base_model_path: Path  # Path to the updated base model to be used in training  
    training_data: Path  # Path to the training dataset  
    params_epochs: int  # Number of epochs for training  
    params_batch_size: int  # Batch size for training  
    params_is_augmentation: bool  # Whether to apply data augmentation during training  
    params_image_size: list  # Image size to be used for training  

# Defining a dataclass to store configuration for evaluating the trained model  
@dataclass(frozen=True)  
class EvaluationConfig:
    path_of_model: Path  # Path to the trained model for evaluation  
    training_data: Path  # Path to the dataset used for evaluation  
    all_params: dict  # Dictionary containing all evaluation parameters  
    mlflow_uri: str  # URI for tracking experiments using MLflow  
    params_image_size: list  # Image size to be used during evaluation  
    params_batch_size: int  # Batch size to be used for evaluation  
