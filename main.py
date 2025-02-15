from cnnClassifier import logger  # Import logger for logging messages
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline  # Import Data Ingestion pipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline  # Import Prepare Base Model pipeline


# Define and execute the Data Ingestion stage
STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")  # Log stage start
   data_ingestion = DataIngestionTrainingPipeline()  # Initialize Data Ingestion pipeline
   data_ingestion.main()  # Execute data ingestion process
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")  # Log stage completion
except Exception as e:
        logger.exception(e)  # Log any exceptions that occur
        raise e  # Re-raise the exception for handling

# Define and execute the Prepare Base Model stage
STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")  # Log separator
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")  # Log stage start
   prepare_base_model = PrepareBaseModelTrainingPipeline()  # Initialize Prepare Base Model pipeline
   prepare_base_model.main()  # Execute base model preparation process
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")  # Log stage completion
except Exception as e:
        logger.exception(e)  # Log any exceptions that occur
        raise e  # Re-raise the exception for handling
