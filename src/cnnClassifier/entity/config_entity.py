from dataclasses import dataclass  # Importing dataclass to create immutable configuration objects
from pathlib import Path  # Importing Path to handle file system paths

# Configuration for Data Ingestion process
@dataclass(frozen=True)  # frozen=True makes instances immutable
class DataIngestionConfig:
    root_dir: Path  # Root directory for data ingestion
    source_URL: str  # URL of the source data
    local_data_file: Path  # Path to store the downloaded data locally
    unzip_dir: Path  # Directory where the data will be extracted

# Configuration for preparing the base model
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path  # Root directory for base model preparation
    base_model_path: Path  # Path to the original base model
    updated_base_model_path: Path  # Path to store the updated base model
    params_image_size: list  # Image size parameter for the model
    params_learning_rate: float  # Learning rate for fine-tuning
    params_include_top: bool  # Whether to include the top layer in the model
    params_weights: str  # Weights initialization type (e.g., 'imagenet')
    params_classes: int  # Number of output classes

# Configuration for training process
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path  # Root directory for training
    trained_model_path: Path  # Path to store the trained model
    updated_base_model_path: Path  # Path to the updated base model
    training_data: Path  # Path to training dataset
    params_epochs: int  # Number of epochs for training
    params_batch_size: int  # Batch size for training
    params_is_augmentation: bool  # Whether data augmentation is applied
    params_image_size: list  # Image size used during training
