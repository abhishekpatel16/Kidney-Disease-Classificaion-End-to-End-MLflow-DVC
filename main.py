# Import the logger for logging messages during execution
from cnnClassifier import logger  

# Import the Data Ingestion pipeline class
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline  

# Define the stage name for logging purposes
STAGE_NAME = "Data Ingestion stage"

try:
    # Log the start of the data ingestion stage
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 

    # Create an instance of the Data Ingestion pipeline
    data_ingestion = DataIngestionTrainingPipeline()

    # Execute the data ingestion pipeline
    data_ingestion.main()

    # Log the successful completion of the data ingestion stage
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    # Log any exceptions that occur during execution
    logger.exception(e)

    # Re-raise the exception to stop execution if an error occurs
    raise e
