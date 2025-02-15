# Import necessary modules from the cnnClassifier package
from cnnClassifier import logger  # Logger for tracking execution details
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline  # Handles data ingestion
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline  # Prepares the base model
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline  # Trains the model
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline  # Evaluates the trained model

# ---------------------- DATA INGESTION STAGE ---------------------- #
STAGE_NAME = "Data Ingestion stage"  # Define the stage name for logging
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")  # Log the start of data ingestion

    # Create an instance of the DataIngestionTrainingPipeline and execute the main method
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")  # Log completion
except Exception as e:
    logger.exception(e)  # Log any exceptions that occur
    raise e  # Raise the exception to halt execution if an error occurs

# ---------------------- PREPARE BASE MODEL STAGE ---------------------- #
STAGE_NAME = "Prepare base model"  # Define the stage name for logging
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")  # Log the start of preparing the base model

    # Create an instance of the PrepareBaseModelTrainingPipeline and execute the main method
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")  # Log completion
except Exception as e:
    logger.exception(e)  # Log any exceptions that occur
    raise e  # Raise the exception to halt execution if an error occurs

# ---------------------- MODEL TRAINING STAGE ---------------------- #
STAGE_NAME = "Training"  # Define the stage name for logging
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")  # Log the start of model training

    # Create an instance of the ModelTrainingPipeline and execute the main method
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")  # Log completion
except Exception as e:
    logger.exception(e)  # Log any exceptions that occur
    raise e  # Raise the exception to halt execution if an error occurs

# ---------------------- MODEL EVALUATION STAGE ---------------------- #
STAGE_NAME = "Evaluation stage"  # Define the stage name for logging
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")  # Log the start of model evaluation

    # Create an instance of the EvaluationPipeline and execute the main method
    model_evalution = EvaluationPipeline()
    model_evalution.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")  # Log completion
except Exception as e:
    logger.exception(e)  # Log any exceptions that occur
    raise e  # Raise the exception to halt execution if an error occurs
