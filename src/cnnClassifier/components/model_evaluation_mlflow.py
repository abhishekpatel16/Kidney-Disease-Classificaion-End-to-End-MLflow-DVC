import tensorflow as tf  # Importing TensorFlow for deep learning operations
from pathlib import Path  # Importing Path to handle file paths
import mlflow  # Importing MLflow for experiment tracking
import mlflow.keras  # Importing MLflow Keras integration for logging models
from urllib.parse import urlparse  # Importing urlparse to parse MLflow URIs
from cnnClassifier.entity.config_entity import EvaluationConfig  # Importing EvaluationConfig dataclass
from cnnClassifier.utils.common import read_yaml, create_directories, save_json  # Importing utility functions

# Evaluation class to handle model evaluation and logging
class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config  # Storing evaluation configuration

    # Private method to create a validation data generator
    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,  # Rescaling pixel values to [0,1]
            validation_split=0.30  # Using 30% of data for validation
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Setting image size
            batch_size=self.config.params_batch_size,  # Setting batch size
            interpolation="bilinear"  # Using bilinear interpolation for resizing
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs  # Applying data generator arguments
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,  # Path to training data
            subset="validation",  # Using validation subset
            shuffle=False,  # Disabling shuffling for consistency
            **dataflow_kwargs  # Applying data flow arguments
        )

    # Static method to load a saved model from a given path
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)  # Loading the Keras model
    
    # Method to perform evaluation
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)  # Loading trained model
        self._valid_generator()  # Creating validation generator
        self.score = self.model.evaluate(self.valid_generator)  # Evaluating model on validation data
        self.save_score()  # Saving evaluation scores

    # Method to save evaluation scores to a JSON file
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}  # Extracting loss and accuracy
        save_json(path=Path("scores.json"), data=scores)  # Saving scores as a JSON file

    # Method to log evaluation results into MLflow
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)  # Setting MLflow registry URI
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme  # Parsing tracking URI
        
        with mlflow.start_run():  # Starting an MLflow run
            mlflow.log_params(self.config.all_params)  # Logging model parameters
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}  # Logging evaluation metrics
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Registering the model in MLflow model registry
                # Additional details can be found in MLflow documentation:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")  # Logging model without registry
