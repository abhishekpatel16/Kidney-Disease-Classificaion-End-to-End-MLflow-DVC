# Import necessary modules
import os  # Provides functions to interact with the operating system
import zipfile  # Allows working with ZIP archives
import gdown  # Used to download files from Google Drive

# Import logger for logging messages
from cnnClassifier import logger  

# Import utility function to get file sizes
from cnnClassifier.utils.common import get_size  

# Import the DataIngestionConfig entity to manage data ingestion configurations
from cnnClassifier.entity.config_entity import DataIngestionConfig  


# Define the DataIngestion class to handle downloading and extracting data
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the DataIngestion class with a given configuration.

        Args:
            config (DataIngestionConfig): Configuration object containing file paths and URLs.
        """
        self.config = config  # Store the configuration object

    
    def download_file(self) -> str:
        '''
        Fetch data from the given URL and save it locally.

        Returns:
            str: Path to the downloaded file.
        '''
        try: 
            # Extract necessary paths from config
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            # Ensure the directory for data ingestion exists
            os.makedirs("artifacts/data_ingestion", exist_ok=True)

            # Log the start of the download
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            # Extract file ID from the Google Drive URL
            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            
            # Download the file from Google Drive
            gdown.download(prefix + file_id, zip_download_dir)

            # Log completion of download
            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            # Raise the exception if an error occurs
            raise e
        
    
    def extract_zip_file(self):
        """
        Extracts the downloaded ZIP file into the specified directory.

        Returns:
            None
        """
        # Get the extraction path from the configuration
        unzip_path = self.config.unzip_dir

        # Ensure the extraction directory exists
        os.makedirs(unzip_path, exist_ok=True)

        # Open and extract the ZIP file
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
