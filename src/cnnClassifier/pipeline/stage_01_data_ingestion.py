# Import ConfigurationManager to manage configurations
from cnnClassifier.config.configuration import ConfigurationManager  

# Import DataIngestion class to handle downloading and extracting data
from cnnClassifier.components.data_ingestion import DataIngestion  

# Import logger to log messages
from cnnClassifier import logger  

# Define a stage name for logging and debugging
STAGE_NAME = "Data Ingestion stage"


# Define the pipeline class for Data Ingestion
class DataIngestionTrainingPipeline:
    def __init__(self):
        """
        Initializes the DataIngestionTrainingPipeline class.
        Currently, no parameters are needed in the constructor.
        """
        pass  

    def main(self):
        """
        The main function to execute the data ingestion pipeline.
        
        Steps:
        1. Load configuration settings.
        2. Retrieve the data ingestion configuration.
        3. Initialize the DataIngestion component with the configuration.
        4. Download the dataset.
        5. Extract the ZIP file.
        """
        # Load the overall configuration
        config = ConfigurationManager()

        # Retrieve the data ingestion-specific configuration
        data_ingestion_config = config.get_data_ingestion_config()

        # Initialize the DataIngestion component with the retrieved configuration
        data_ingestion = DataIngestion(config=data_ingestion_config)

        # Download the dataset from the source URL
        data_ingestion.download_file()

        # Extract the ZIP file to the specified directory
        data_ingestion.extract_zip_file()


# Run the data ingestion pipeline if this script is executed directly
if __name__ == '__main__':
    try:
        # Log the start of the data ingestion stage
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        # Create an instance of the pipeline and run the main function
        obj = DataIngestionTrainingPipeline()
        obj.main()

        # Log the completion of the data ingestion stage
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        # Log any exceptions that occur and raise the error
        logger.exception(e)
        raise e
