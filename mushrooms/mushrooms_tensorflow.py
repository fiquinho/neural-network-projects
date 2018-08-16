import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

SCRIPT_DIR = Path.cwd()
sys.path.append(str(SCRIPT_DIR.parent))

from mushrooms.process_data import data_preparation
from data_utils.data_iterator import DataIterator
from tensorflow_base_models.base_tf_model import BaseTensorflowModel


# Logging operations
logger = logging.getLogger()
logger.setLevel(logging.INFO)
FORMATTER = logging.Formatter('%(asctime)s: %(filename)s - [ %(message)s ]',
                              '%m/%d/%Y %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(FORMATTER)
logger.addHandler(console)


class MushroomsTensorflowModel(BaseTensorflowModel):
    """
    Class used to create a model for the mushrooms classification task.
    """

    def __init__(self, data_features: int, fully_connected_layers: int, fully_connected_size: int,
                 fully_connected_activation: str):
        """
        Create a instance of the class. Each instance can train a different model architecture.

        :param data_features: The number of features of a single data point.
        :param fully_connected_layers: The number of hidden fully connected layers of the model.
        :param fully_connected_size: The number of hidden units of the fully connected layers.
        :param fully_connected_activation: The activation function of the fully connected layers.
        """
        super(MushroomsTensorflowModel, self).__init__()

        self.model_name = "Mushrooms_Tensorflow_Model"

        # Model parameters
        self.data_features = data_features
        self.fully_connected_layers = fully_connected_layers
        self.fully_connected_size = fully_connected_size

        if fully_connected_activation == "relu":
            self.fully_connected_activation = tf.nn.relu
        elif fully_connected_activation == "sigmoid":
            self.fully_connected_activation = tf.nn.sigmoid
        else:
            raise ValueError("The activation function {} is not in the available "
                             "activation functions.".format(fully_connected_activation))

    def _create_placeholders(self):
        """
        Create the placeholders of the model.
        """
        logger.info("Creating model placeholders.")

        self.inputs_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self.data_features], name="inputs")
        logger.info(self.inputs_placeholder)
        self.labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="labels")
        logger.info(self.labels_placeholder)

        self.learning_rate_input = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")
        logger.info(self.learning_rate_input)

    def _build_forward(self):
        """
        Create the tensorflow graph for this instance's model.
        """

        logger.info("Building model graph.")

        dense_layers = self.fully_connected_layers

        # Create first dense layer
        # Shape: (batch_size, fully_connected_size)
        dense = tf.layers.dense(self.inputs_placeholder,
                                units=self.fully_connected_size,
                                activation=self.fully_connected_activation,
                                name="dense_1")

        logger.info(dense)

        dense_layers -= 1
        dense_layer_count = 2

        # Create next dense layers
        for i in range(dense_layers):
            # Shape: (batch_size, fully_connected_size)
            dense = tf.layers.dense(dense,
                                    units=self.fully_connected_size,
                                    activation=self.fully_connected_activation,
                                    name="dense_{}".format(dense_layer_count))

            logger.info(dense)

            dense_layer_count += 1

        # Create output layer
        # Shape: (batch_size, 1)
        self.output_layer = tf.layers.dense(dense,
                                            units=1,
                                            activation=tf.nn.sigmoid,
                                            name="output")

        logger.info(self.output_layer)

        with tf.variable_scope("cost_and_optimizer"):
            # Loss value for the model output of each data point fed to the graph
            # Shape: (batch_size, 1)
            self.loss_op = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels_placeholder,
                                                                   logits=self.output_layer,
                                                                   name="loss_op")
            logger.info(self.loss_op)

            # Cost value for all the data points fed to the model
            # Shape: ()
            self.cost = tf.reduce_sum(self.loss_op, name="cost")
            logger.info(self.cost)

            # Model optimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_input,
                                                          name="gradient_descent_optimizer")

            # Training operation. Perform one parameters update. Increment global_step by 1.
            self.training_op = optimizer.minimize(self.cost, global_step=self.global_step)

        # The model outputs a value between 0 and 1. We need to transform
        # it into the predicted label (0 or 1)
        predictions = tf.round(self.output_layer, name="predictions")

        # Metrics
        with tf.name_scope("metrics"):
            # Get the model's accuracy.
            self.accuracy = tf.metrics.accuracy(labels=self.labels_placeholder,
                                                predictions=predictions,
                                                name="accuracy")

        # Tensorboard operations
        with tf.name_scope("summaries"):
            tf.summary.scalar("cost", self.cost)
            tf.summary.scalar("accuracy", self.accuracy[0])
            tf.summary.histogram("output_layer", self.output_layer)
            self.summary_op = tf.summary.merge_all()

        self.tensorboard_job_name = "tensorflow_dl_{}-ds_{}".format(self.fully_connected_layers,
                                                                    self.fully_connected_size)

    def _get_training_feed_dict(self, training_data: np.array, training_labels: np.array,
                                learning_rate: float) -> dict:
        """
        Generate a dictionary to feed the model's graph for training.

        :param training_data: A batch of data points to train the model with.
        :param training_labels: A batch of labels of the data points.
        :param learning_rate: The learning rate to use during training.
        :return: The feeding dictionary for the model's graph.
        """
        feed_dict = {self.inputs_placeholder: training_data,
                     self.labels_placeholder: training_labels,
                     self.learning_rate_input: learning_rate}
        return feed_dict

    def _get_validation_feed_dict(self, validation_data, validation_labels) -> dict:
        """
        Generate a dictionary to feed the model's graph for evaluating the model.

        :param validation_data: A batch of data point to evaluate the model on.
        :param validation_labels: A batch of labels of the data points.
        :return: The feeding dictionary for the model's graph.
        """
        feed_dict = {self.inputs_placeholder: validation_data,
                     self.labels_placeholder: validation_labels}
        return feed_dict


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

    # Data iterators
    training_iterator = DataIterator(data=training[0], labels=training[1],
                                     batch_size=args.batch_size, shuffle=True)
    validation_iterator = DataIterator(data=validation[0], labels=validation[1],
                                       batch_size=args.batch_size, shuffle=True)

    # Create model instance, build the graph and train on the provided data
    model = MushroomsTensorflowModel(data_features=training[0].shape[1],
                                     fully_connected_layers=args.dense_layers,
                                     fully_connected_size=args.dense_size,
                                     fully_connected_activation=args.dense_activation)
    model.build_graph()
    model.train_model(epochs=args.epochs,
                      learning_rate=args.learning_rate,
                      val_period=args.val_period,
                      save_period=args.save_period,
                      training_iterator=training_iterator,
                      validation_iterator=validation_iterator)

    # Plot training information
    if not args.no_plots:
        model.plot_train_stats()


if __name__ == '__main__':
    main()
