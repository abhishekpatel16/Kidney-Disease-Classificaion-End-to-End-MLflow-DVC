# Import necessary libraries
import os  # Provides functions for interacting with the operating system
import urllib.request as request  # Enables downloading files from the internet
from zipfile import ZipFile  # Helps extract ZIP archive files
import tensorflow as tf  # Deep learning library for building and training neural networks
import time  # Used for tracking execution time
from pathlib import Path  # Provides an object-oriented way to handle file paths
from cnnClassifier.entity.config_entity import TrainingConfig  # Imports TrainingConfig, which contains configuration settings

# Define the Training class
class Training:
    def __init__(self, config: TrainingConfig):
        """
        Initializes the Training class with configuration settings.
        
        Args:
            config (TrainingConfig): Configuration object containing training parameters.
        """
        self.config = config

    def get_base_model(self):
        """
        Loads a pre-trained model from the specified path in the configuration.
        """
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path  # Path to the pre-trained base model
        )

    def train_valid_generator(self):
        """
        Creates training and validation data generators for image augmentation and processing.
        """

        # Common preprocessing parameters for both training and validation datasets
        datagenerator_kwargs = dict(
            rescale=1. / 255,  # Normalize pixel values to range [0, 1]
            validation_split=0.20  # Use 20% of data for validation
        )

        # Parameters for resizing and batch processing images
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Resize images to required dimensions
            batch_size=self.config.params_batch_size,  # Batch size for training
            interpolation="bilinear"  # Interpolation method for resizing
        )

        # Create a validation data generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Generate validation dataset
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,  # Path to training data directory
            subset="validation",  # Use this subset for validation
            shuffle=False,  # Don't shuffle validation data
            **dataflow_kwargs
        )

        # Check if data augmentation is enabled in the configuration
        if self.config.params_is_augmentation:
            # Apply various augmentation techniques for training data
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,  # Rotate images up to 40 degrees
                horizontal_flip=True,  # Flip images horizontally
                width_shift_range=0.2,  # Shift images horizontally by up to 20%
                height_shift_range=0.2,  # Shift images vertically by up to 20%
                shear_range=0.2,  # Apply shear transformations
                zoom_range=0.2,  # Random zoom-in/out
                **datagenerator_kwargs  # Include common preprocessing parameters
            )
        else:
            # Use the same generator for training and validation (no augmentation)
            train_datagenerator = valid_datagenerator

        # Generate training dataset
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,  # Path to training data directory
            subset="training",  # Use this subset for training
            shuffle=True,  # Shuffle training data for better learning
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the trained model to the specified path.
        
        Args:
            path (Path): Destination path for saving the model.
            model (tf.keras.Model): The trained model instance.
        """
        model.save(path)  # Save the model

    def train(self):
        """
        Trains the model using the prepared training and validation datasets.
        """

        # Calculate the number of training steps per epoch
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        # Calculate the number of validation steps per epoch
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model
        self.model.fit(
            self.train_generator,  # Training dataset
            epochs=self.config.params_epochs,  # Number of epochs from configuration
            steps_per_epoch=self.steps_per_epoch,  # Steps per epoch
            validation_steps=self.validation_steps,  # Steps for validation
            validation_data=self.valid_generator  # Validation dataset
        )

        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,  # Path where the trained model will be stored
            model=self.model  # Trained model instance
        )
