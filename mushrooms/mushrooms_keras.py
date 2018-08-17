import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

SCRIPT_DIR = Path.cwd()
sys.path.append(str(SCRIPT_DIR.parent))

from mushrooms.process_data import data_preparation


# Logging operations
logger = logging.getLogger()
logger.setLevel(logging.INFO)
FORMATTER = logging.Formatter('%(asctime)s: %(filename)s - [ %(message)s ]',
                              '%m/%d/%Y %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(FORMATTER)
logger.addHandler(console)


class MushroomsKerasModel(object):
    """
    Class used to create a model for the mushrooms classification task.
    """

    def __init__(self, data_features: int, fully_connected_layers: int, fully_connected_size: int,
                 fully_connected_activation: str, epochs: int, batch_size: int,
                 learning_rate: float):
        """
        Create a instance of the class. Each instance can train a different model architecture.

        :param data_features: The number of features of a single data point.
        :param fully_connected_layers: The number of hidden fully connected layers of the model.
        :param fully_connected_size: The number of hidden units of the fully connected layers.
        :param fully_connected_activation: The activation function of the fully connected layers.
        :param epochs: Number of training runs over the entire data.
        :param batch_size: How many data points to feed to the model for each training step.
        :param learning_rate: The scaling factor to use on the gradients on each training step.
        """

        self.model_name = "Mushrooms_Tensorflow_Model"

        # Model parameters
        self.data_features = data_features
        self.fully_connected_layers = fully_connected_layers
        self.fully_connected_size = fully_connected_size
        self.fully_connected_activation = fully_connected_activation

        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Model variables
        self.model = None
        self.input = None
        self.output_layer = None
        self.callbacks = None
        self.history = None

    def build_model(self):
        """
        Create the Keras model for this instance.
        """

        logger.info("Building model graph.")

        # Data input
        self.input = tf.keras.Input(shape=(self.data_features,), name="inputs")
        logger.info(self.input)

        dense_layers = self.fully_connected_layers

        # Create first dense layer
        # Shape: (batch_size, fully_connected_size)
        dense = tf.keras.layers.Dense(self.fully_connected_size,
                                      activation=self.fully_connected_activation,
                                      name="dense_1")(self.input)

        logger.info(dense)

        dense_layers -= 1
        dense_layer_count = 2

        # Create next dense layers
        for i in range(dense_layers):
            # Shape: (batch_size, fully_connected_size)
            dense = tf.keras.layers.Dense(self.fully_connected_size,
                                          activation=self.fully_connected_activation,
                                          name="dense_{}".format(dense_layer_count))(dense)

            logger.info(dense)

            dense_layer_count += 1

        # Create output layer
        # Shape: (batch_size, 1)
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid",
                                                  name="output_layer")(dense)

        logger.info(self.output_layer)

        self.model = tf.keras.Model(inputs=self.input, outputs=self.output_layer)

        self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=self.learning_rate),
                           loss="binary_crossentropy",
                           metrics=["accuracy"])

        self.model.summary()

        # Create callbacks
        self.callbacks = []

        # Model checkpoint (save model)
        model_dir = Path(Path.cwd(), "trained_models", "keras")
        model_dir.mkdir(exist_ok=True, parents=True)
        model_file = "./trained_models/keras/model.hdf5"
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file, monitor='val_acc', save_best_only=True)
        self.callbacks.append(model_checkpoint)

        # File with training history (.csv)
        csv_logger = tf.keras.callbacks.CSVLogger(str(model_dir) + "/training_history.csv")
        self.callbacks.append(csv_logger)

        # Tensorboard
        tensorboard_logs_dir = Path(Path.cwd(), "tensorboard_logs")
        tensorboard_logs_dir.mkdir(exist_ok=True)
        tensorboard_job_name = "keras_dl_{}-ds_{}-lr_{}-e_{}-b_{}".format(self.fully_connected_layers,
                                                                          self.fully_connected_size,
                                                                          self.learning_rate,
                                                                          self.epochs,
                                                                          self.batch_size)
        tensorboard_log_dir = Path(tensorboard_logs_dir, tensorboard_job_name)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_log_dir),
                                                              histogram_freq=1,
                                                              write_graph=False,
                                                              write_grads=True,
                                                              write_images=True,
                                                              batch_size=self.batch_size)
        self.callbacks.append(tensorboard_callback)

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_acc",
                                                          patience=10)
        self.callbacks.append(early_stopping)

        writer = tf.summary.FileWriter(str(tensorboard_log_dir))
        writer.add_graph(tf.get_default_graph())

    def train_model(self, training_data: np.array, training_labels: np.array,
                    validation_data: np.array, validation_labels: np.array):
        """
        Train this instance's model.

        :param training_data: The full training data points (data_instances, data_features).
        :param training_labels: The full labels for the data points (data_instances, 1).
        :param validation_data: The full validation data points (validation_instances, data_features).
        :param validation_labels: The full labels for the validation points (validation_instances, 1).
        """

        self.history = self.model.fit(training_data, training_labels,
                                      epochs=self.epochs,
                                      validation_data=(validation_data, validation_labels),
                                      verbose=2,
                                      batch_size=self.batch_size,
                                      callbacks=self.callbacks,
                                      shuffle=True)

    def plot_training_information(self):
        """
        Plot training information. Generates a figure with 4 plots:
         - The cost during training.
         - The accuracy during training.
         - The cost on the validation data, during training.
         - The accuracy on the validation data, during training.
        """

        # Generate figure with 4 plots
        f, subplots = plt.subplots(2, 2)

        # Training cost plot
        subplots[0, 0].plot(range(len(self.history.history["loss"])),
                            self.history.history["loss"], color="blue")
        subplots[0, 0].set_title("Training cost")
        subplots[0, 0].grid()

        # Training accuracy plot
        subplots[0, 1].plot(range(len(self.history.history["val_loss"])),
                            self.history.history["val_loss"], color="red")
        subplots[0, 1].set_title("Validation cost")
        subplots[0, 1].grid()

        # Validation cost plot
        subplots[1, 0].plot(range(len(self.history.history["acc"])),
                            self.history.history["acc"], color="blue")
        subplots[1, 0].set_title("Training accuracy")
        subplots[1, 0].grid()

        # Validation accuracy plot
        subplots[1, 1].plot(range(len(self.history.history["val_acc"])),
                            self.history.history["val_acc"], color="red")
        subplots[1, 1].set_title("Validation accuracy")
        subplots[1, 1].grid()

        # Space between plots
        f.subplots_adjust(hspace=0.3)

        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Create and train a neural network model for the "
                                                 "mushroom classification task.")

    # Optional arguments
    parser.add_argument("--no_plots", default=False, action="store_true",
                        help="Activate to avoid generating plots after training.")
    parser.add_argument("--split", type=float, default=0.1,
                        help="The portion of the data to use as validation and test sets. "
                             "Defaults to 0.1.")
    parser.add_argument("--dense_layers", type=int, default=3,
                        help="The number of fully connected layers in the model. Defaults to 3.")
    parser.add_argument("--dense_size", type=int, default=50,
                        help="The size of the fully connected layers. Defaults to 50.")
    parser.add_argument("--dense_activation", type=str, default="relu", choices=["relu", "sigmoid"],
                        help="The activation of the fully connected layers. Defaults to 'relu'.")
    parser.add_argument("--epochs", type=int, default=25,
                        help="The number of epochs to perform during training. Defaults to 25.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="The batch size to use during training. Defaults to 128.")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="The learning rate to use during training. Defaults to 0.01.")
    parser.add_argument("--val_period", type=int, default=10,
                        help="Number of training steps between model evaluations on the "
                             "validation data. Defaults to 10.")
    parser.add_argument("--save_period", type=int, default=100,
                        help="Number of training steps between model checkpoints. Defaults to 100.")
    args = parser.parse_args()

    # Process the data
    training, validation, test = data_preparation(data_split=args.split)

    # Create model instance, build the graph and train on the provided data
    model = MushroomsKerasModel(data_features=training[0].shape[1],
                                fully_connected_layers=args.dense_layers,
                                fully_connected_size=args.dense_size,
                                fully_connected_activation=args.dense_activation,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                learning_rate=args.learning_rate)
    model.build_model()
    model.train_model(training_data=training[0],
                      training_labels=training[1],
                      validation_data=validation[0],
                      validation_labels=validation[1])

    # Plot training information
    if not args.no_plots:
        model.plot_training_information()


if __name__ == '__main__':
    main()
