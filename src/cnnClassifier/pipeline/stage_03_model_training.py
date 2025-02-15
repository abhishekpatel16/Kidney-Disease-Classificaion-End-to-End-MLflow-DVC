# Import necessary modules from the cnnClassifier package
from cnnClassifier.config.configuration import ConfigurationManager  # Manages configurations for model training
from cnnClassifier.components.model_training import Training  # Handles model training operations
from cnnClassifier import logger  # Logger for logging messages

# Define the stage name for logging purposes
STAGE_NAME = "Training"


# Define the ModelTrainingPipeline class
class ModelTrainingPipeline:
    def __init__(self):
        """
        Initializes the ModelTrainingPipeline class.
        Currently, no initialization parameters are required.
        """
        pass  # No initialization required at this stage

    def main(self):
        """
        Main method to execute the model training pipeline.
        """
        # Create an instance of ConfigurationManager to fetch configuration details
        config = ConfigurationManager()
        
        # Retrieve training configuration settings
        training_config = config.get_training_config()
        
        # Initialize the Training class with the retrieved configuration
        training = Training(config=training_config)
        
        # Load the base model (pre-trained model)
        training.get_base_model()
        
        # Prepare training and validation data generators
        training.train_valid_generator()
        
        # Train the model using the prepared data
        training.train()


# Execute the script only if it is run as the main module
if __name__ == '__main__':
    try:
        # Log the start of the training stage
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Create an instance of ModelTrainingPipeline
        obj = ModelTrainingPipeline()
        
        # Run the main training pipeline
        obj.main()
        
        # Log the completion of the training stage
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    
    except Exception as e:
        # Log any exceptions that occur during execution
        logger.exception(e)
        
        # Raise the exception to halt execution
        raise e
