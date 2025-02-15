from cnnClassifier.config.configuration import ConfigurationManager  # Import ConfigurationManager to manage configurations
from cnnClassifier.components.prepare_base_model import PrepareBaseModel  # Import PrepareBaseModel class to prepare the base model
from cnnClassifier import logger  # Import logger for logging messages

# Define the stage name for logging purposes
STAGE_NAME = "Prepare base model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        """
        Initialize the PrepareBaseModelTrainingPipeline class.
        """
        pass

    def main(self):
        """
        Executes the main process of preparing the base model.
        """
        config = ConfigurationManager()  # Create a ConfigurationManager instance
        prepare_base_model_config = config.get_prepare_base_model_config()  # Fetch the base model configuration
        
        # Initialize PrepareBaseModel with the retrieved configuration
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        
        # Load and save the base model
        prepare_base_model.get_base_model()
        
        # Update and save the modified base model
        prepare_base_model.update_base_model()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")  # Log the start of the process
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Create an instance of the training pipeline and execute the main function
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")  # Log the completion of the process
    except Exception as e:
        logger.exception(e)  # Log any exceptions that occur
        raise e  # Re-raise the exception for further handling
