# Import necessary modules from the cnnClassifier package
from cnnClassifier import logger  # Logger for logging messages
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline  # Handles data ingestion
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline  # Prepares the base model
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline  # Handles model training

# Define the stage name for data ingestion
STAGE_NAME = "Data Ingestion stage"
try:
   # Log the start of the data ingestion stage
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   
   # Create an instance of DataIngestionTrainingPipeline
   data_ingestion = DataIngestionTrainingPipeline()
   
   # Run the data ingestion process
   data_ingestion.main()
   
   # Log the completion of the data ingestion stage
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   # Log any exceptions that occur during execution
   logger.exception(e)
   
   # Raise the exception to halt execution
   raise e

# Define the stage name for preparing the base model
STAGE_NAME = "Prepare base model"
try: 
   # Log the start of the base model preparation stage
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   
   # Create an instance of PrepareBaseModelTrainingPipeline
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   
   # Run the base model preparation process
   prepare_base_model.main()
   
   # Log the completion of the base model preparation stage
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   # Log any exceptions that occur during execution
   logger.exception(e)
   
   # Raise the exception to halt execution
   raise e

# Define the stage name for training the model
STAGE_NAME = "Training"
try: 
   # Log the start of the training stage
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   
   # Create an instance of ModelTrainingPipeline
   model_trainer = ModelTrainingPipeline()
   
   # Run the model training process
   model_trainer.main()
   
   # Log the completion of the training stage
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   # Log any exceptions that occur during execution
   logger.exception(e)
   
   # Raise the exception to halt execution
   raise e
