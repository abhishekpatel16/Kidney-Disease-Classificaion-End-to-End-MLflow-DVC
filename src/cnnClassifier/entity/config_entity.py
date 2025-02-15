# Import necessary modules
from dataclasses import dataclass  # Provides a decorator to define a data class
from pathlib import Path  # Pathlib provides an object-oriented approach to handle file paths

# Define a data class for data ingestion configuration
@dataclass(frozen=True)  # `frozen=True` makes the class immutable (prevents modifications after creation)
class DataIngestionConfig:
    root_dir: Path  # Path to the root directory where data will be stored
    source_URL: str  # URL of the dataset to be downloaded
    local_data_file: Path  # Path where the downloaded data file will be saved
    unzip_dir: Path  # Directory where the dataset will be extracted



@dataclass(frozen=True)  # `frozen=True` makes the dataclass immutable (prevents modifications after initialization)
class PrepareBaseModelConfig:
    """
    Configuration class for preparing the base model.

    Attributes:
        root_dir (Path): Directory where the base model-related artifacts will be stored.
        base_model_path (Path): Path to the initial pre-trained base model.
        updated_base_model_path (Path): Path where the updated base model will be saved.
        params_image_size (list): Input image dimensions (e.g., [224, 224, 3] for VGG16).
        params_learning_rate (float): Learning rate for model training.
        params_include_top (bool): Whether to include the top classification layers in the base model.
        params_weights (str): Pre-trained weights to use (e.g., "imagenet").
        params_classes (int): Number of output classes for classification.
    """
    
    root_dir: Path  # Root directory for storing model-related files
    base_model_path: Path  # Path to the base model file
    updated_base_model_path: Path  # Path where the updated model will be stored
    params_image_size: list  # Image input size for the model
    params_learning_rate: float  # Learning rate for model training
    params_include_top: bool  # Whether to include the top layers of the model
    params_weights: str  # Pre-trained model weights (e.g., "imagenet")
    params_classes: int  # Number of output classes