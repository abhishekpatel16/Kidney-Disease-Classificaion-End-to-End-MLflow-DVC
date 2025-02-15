# Import necessary modules from the cnnClassifier package
from cnnClassifier.config.configuration import ConfigurationManager  # Manages configuration settings
from cnnClassifier.components.model_evaluation_mlflow import Evaluation  # Handles model evaluation
from cnnClassifier import logger  # Logger for tracking execution details

# Define the stage name for logging purposes
STAGE_NAME = "Evaluation stage"

# Define a class for the evaluation pipeline
class EvaluationPipeline:
    def __init__(self):
        pass  # No initialization needed for this class

    def main(self):
        """
        Main function to execute the evaluation process.
        """
        config = ConfigurationManager()  # Create an instance of ConfigurationManager to get configurations
        eval_config = config.get_evaluation_config()  # Retrieve evaluation-specific configurations
        evaluation = Evaluation(eval_config)  # Create an Evaluation object with the configuration
        evaluation.evaluation()  # Perform model evaluation
        evaluation.save_score()  # Save the evaluation score
        # evaluation.log_into_mlflow()  # Log the results into MLflow (commented out, can be enabled if needed)

# Main execution block
if __name__ == '__main__':
    try:
        # Log the start of the evaluation stage
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        # Create an instance of the EvaluationPipeline class and run the main method
        obj = EvaluationPipeline()
        obj.main()

        # Log the completion of the evaluation stage
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        # Log any exceptions that occur during execution
        logger.exception(e)
        raise e  # Re-raise the exception to halt execution if an error occurs
