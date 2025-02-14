import os # Importing the os module for interacting with the operating system
from box.exceptions import BoxValueError  # Exception handling for Box
import yaml  # YAML file handling
from cnnClassifier import logger  # Logger for logging messages
import json  # JSON file handling
import joblib  # Serialization and deserialization of objects
from ensure import ensure_annotations  # Ensures type annotations are respected
from box import ConfigBox  # A dictionary-like object with dot-accessible attributes
from pathlib import Path  # Path handling for files and directories
from typing import Any  # Type hinting for functions
import base64  # Encoding and decoding binary data

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its content as a ConfigBox object.
    
    Args:
        path_to_yaml (Path): Path to the YAML file.
    
    Raises:
        ValueError: If the YAML file is empty.
        Exception: Any other exceptions that may occur.
    
    Returns:
        ConfigBox: YAML content wrapped in a ConfigBox.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Creates multiple directories if they don't exist.
    
    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool, optional): If True, logs the directory creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves a dictionary as a JSON file.
    
    Args:
        path (Path): Path where the JSON file should be saved.
        data (dict): Dictionary containing the data to be saved.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads JSON file data and returns it as a ConfigBox object.
    
    Args:
        path (Path): Path to the JSON file.
    
    Returns:
        ConfigBox: JSON content wrapped in a ConfigBox.
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """Saves data as a binary file using joblib.
    
    Args:
        data (Any): Data to be saved.
        path (Path): Path where the binary file should be saved.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads binary data from a file using joblib.
    
    Args:
        path (Path): Path to the binary file.
    
    Returns:
        Any: Loaded binary data.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """Gets the size of a file in KB.
    
    Args:
        path (Path): Path to the file.
    
    Returns:
        str: File size in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"

def decodeImage(imgstring, fileName):
    """Decodes a base64 encoded image and saves it to a file.
    
    Args:
        imgstring (str): Base64 encoded image data.
        fileName (str): Filename where the decoded image should be saved.
    """
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)

def encodeImageIntoBase64(croppedImagePath):
    """Encodes an image file into base64 format.
    
    Args:
        croppedImagePath (str): Path to the image file.
    
    Returns:
        str: Base64 encoded string of the image.
    """
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
