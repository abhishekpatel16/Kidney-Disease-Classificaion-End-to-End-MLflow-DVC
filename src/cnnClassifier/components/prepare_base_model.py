import os  # Importing OS module for file operations
import urllib.request as request  # Importing request module to handle URL requests
from zipfile import ZipFile  # Importing ZipFile to extract zip files
import tensorflow as tf  # Importing TensorFlow for deep learning operations
from pathlib import Path  # Importing Path from pathlib for file path management
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig  # Importing custom configuration entity

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initializes the PrepareBaseModel class with configuration settings.
        :param config: Configuration object containing model parameters and file paths.
        """
        self.config = config
    
    def get_base_model(self):
        """
        Loads the VGG16 base model with specified parameters and saves it.
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,  # Set input image size
            weights=self.config.params_weights,  # Load pre-trained weights
            include_top=self.config.params_include_top  # Exclude fully connected layers if needed
        )
        
        # Save the loaded base model
        self.save_model(path=self.config.base_model_path, model=self.model)
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepares a modified version of the base model by adding custom layers.
        :param model: Base model
        :param classes: Number of output classes
        :param freeze_all: If True, freezes all layers of the model
        :param freeze_till: If set, freezes layers up to a specified index
        :param learning_rate: Learning rate for the optimizer
        :return: Modified full model
        """
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False  # Freeze all layers
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False  # Freeze layers up to freeze_till index

        # Add a flattening layer
        flatten_in = tf.keras.layers.Flatten()(model.output)
        
        # Add a dense output layer with softmax activation
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        # Create the full model
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Compile the model
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()  # Print model summary
        return full_model
    
    def update_base_model(self):
        """
        Updates the base model by adding new layers and saving the updated model.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,  # Freezing all layers of the base model
            freeze_till=None,  # No selective freezing
            learning_rate=self.config.params_learning_rate
        )
        
        # Save the updated model
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the given model to the specified path.
        :param path: File path where the model will be saved
        :param model: TensorFlow Keras model to save
        """
        model.save(path)
