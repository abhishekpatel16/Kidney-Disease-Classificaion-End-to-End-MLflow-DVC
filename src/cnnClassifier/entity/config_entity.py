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
