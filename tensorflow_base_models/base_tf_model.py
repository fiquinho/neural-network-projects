"""
This script has classes with base Tensorflow models. They can be used to
reuse functionality (like training) between different model architectures,
for different projects.
"""

# TODO: Add option to build the model for Eager execution mode.

import logging
import sys
from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path.cwd()
sys.path.append(str(SCRIPT_DIR.parent))

from data_utils.data_iterator import DataIterator


logger = logging.getLogger()


class BaseTensorflowModel(object):
    """
    Class used as a base Tensorflow model that provides many reusable functions for different
    model architectures. Classes that inherit from this will need to define the functions:
     - _create_placeholders
     - _build_forward
     - _get_training_feed_dict
     - _get_validation_feed_dict
    """

    def __init__(self):

        self.model_name = None
        self.generated_graph = False
        self.global_step = tf.get_variable(name="global_step",
                                           shape=[],
                                           dtype='int32',
                                           initializer=tf.constant_initializer(0),
                                           trainable=False)

        self.output_layer = None
        self.loss_op = None
        self.cost = None
        self.accuracy = None

        self.training_op = None
        self.summary_op = None

        self.tensorboard_job_name = None
        self.tensorboard_log_dir = None

        self.training_costs = None
        self.training_accuracies = None
        self.validation_costs = None
        self.validation_accuracies = None

    def _create_placeholders(self):
        raise NotImplementedError

    def _build_forward(self):
        raise NotImplementedError

    def _get_training_feed_dict(self, training_data, training_labels, learning_rate):
        raise NotImplementedError

    def _get_validation_feed_dict(self, validation_data, validation_labels):
        raise NotImplementedError

    def build_graph(self):
        """
        Build the model's graph. The functions '_create_placeholders' and '_build_forward'
        need to be defined in the child class of this one.
        """
        logger.info("Building tensorflow graph for {}.".format(self.model_name))
        self._create_placeholders()
        self._build_forward()

        self.generated_graph = True

    def train_model(self, epochs: int, learning_rate: float,
                    val_period: int, save_period: int, training_iterator: DataIterator,
                    validation_iterator: DataIterator):
        """
        Train the generated model of this class instance.

        :param epochs: Number of training runs over the entire data.
        :param learning_rate: The scaling factor to use on the gradients on each training step.
        :param val_period: How many training steps between model evaluations on the validation data.
        :param save_period: How many training steps between model checkpoints.
        :param training_iterator: A batch generator of training data and labels.
        :param validation_iterator: A batch generator of validation data and labels.
        """

        # Check if the model's graph is already generated
        if not self.generated_graph:
            raise Exception("The model graph is not generated yet. Use the function 'build_graph' "
                            "before trying to perform the training.")

        global epoch

        # Tensorboard directory and file name generation
        tensorboard_logs_dir = Path(Path.cwd(), "tensorboard_logs")
        tensorboard_logs_dir.mkdir(exist_ok=True)
        self.tensorboard_job_name += "-lr_{}-e_{}-b_{}".format(learning_rate, epochs, training_iterator.batch_size)
        self.tensorboard_log_dir = Path(tensorboard_logs_dir, self.tensorboard_job_name)

        # This tensorflow operations are used to initialize all the variables of the model.
        init_op = tf.global_variables_initializer()
        init_l_op = tf.local_variables_initializer()

        with tf.Session() as sess:
            # Initialize graph variables
            sess.run(init_op)
            sess.run(init_l_op)

            # Training and validation Tensorboard files writers
            train_writer = tf.summary.FileWriter(str(Path(self.tensorboard_log_dir, "train")),
                                                 sess.graph)
            validation_writer = tf.summary.FileWriter(str(Path(self.tensorboard_log_dir, "validation")),
                                                      sess.graph)

            # Set up a Saver for periodically serializing the model
            saver = tf.train.Saver(max_to_keep=5)

            # Create lists to store useful data
            self.training_costs = []
            self.training_accuracies = []
            self.validation_costs = []
            self.validation_accuracies = []

            # Iterate over the entire data
            for epoch in range(epochs):

                # Iterate over a generator that returns batches
                for train_batch in training_iterator:

                    # Keep track of the number of training steps performed so far
                    global_step_count = sess.run(self.global_step)

                    # Generate training dictionary to feed the model
                    feed_dict = self._get_training_feed_dict(training_data=train_batch[0],
                                                             training_labels=train_batch[1],
                                                             learning_rate=learning_rate)

                    # Do a gradient update, log results to Tensorboard and save information to lists
                    train_cost, train_accuracy, _, train_summary = sess.run(
                        [self.cost, self.accuracy, self.training_op, self.summary_op],
                        feed_dict=feed_dict)
                    train_writer.add_summary(train_summary, global_step_count)

                    self.training_costs.append(train_cost)
                    self.training_accuracies.append(train_accuracy[0])

                    # Evaluate the model on validation set every val_period steps
                    if global_step_count % val_period == 0:
                        mean_val_accuracy, mean_val_cost, val_summary = self._evaluate_on_validation(
                            validation_iterator=validation_iterator, session=sess)

                        self.validation_costs.append(mean_val_cost)
                        self.validation_accuracies.append(mean_val_accuracy)
                        validation_writer.add_summary(val_summary, global_step_count)

                    # Write a model checkpoint if necessary.
                    if global_step_count % save_period == 0:
                        Path(Path.cwd(), "trained_models", "tensorflow").mkdir(exist_ok=True, parents=True)
                        saver.save(sess, "./trained_models/tensorflow/model.ckpt")

                # At the end of each epoch evaluate the model
                mean_val_accuracy, mean_val_cost, val_summary = self._evaluate_on_validation(
                    validation_iterator=validation_iterator, session=sess)

                self.validation_costs.append(mean_val_cost)
                self.validation_accuracies.append(mean_val_accuracy)
                validation_writer.add_summary(val_summary, global_step_count)

                # Print information about training
                logger.info("Epoch {} - Global step {} - Training cost {} - Training accuracy {} - "
                            "Validation cost {} - Validation accuracy {}".format(epoch,
                                                                                 global_step_count,
                                                                                 self.training_costs[-1],
                                                                                 self.training_accuracies[-1],
                                                                                 mean_val_cost,
                                                                                 mean_val_accuracy))

            # Done training!
            logger.info("Finished {} epochs!".format(epoch + 1))
            Path(Path.cwd(), "trained_models", "tensorflow").mkdir(exist_ok=True, parents=True)
            save_path = saver.save(sess, "./trained_models/tensorflow/model.ckpt")
            logger.info("Model saved in path: {}".format(save_path))

    def _evaluate_on_validation(self, validation_iterator: DataIterator,
                                session: tf.Session()) -> (float, float, tf.Summary):
        """
        Evaluate the model on validation data, returned by a batch generator.

        :param validation_iterator: A batch generator of validation data and labels.
        :param session: The tensorflow session to use to perform the evaluation.
        :return: A tuple with the calculated accuracy, cost and a summary to add to a Tensorboard file.
        """

        # Calculate the mean of the validation metrics
        # over the validation set.
        val_batch_accuracies = []
        val_batch_costs = []
        for val_batch in validation_iterator:
            feed_dict = self._get_validation_feed_dict(validation_data=val_batch[0],
                                                       validation_labels=val_batch[1])
            val_batch_acc, val_batch_cost = session.run([self.accuracy, self.cost],
                                                        feed_dict=feed_dict)

            val_batch_accuracies.append(val_batch_acc)
            val_batch_costs.append(val_batch_cost)

        # Take the mean of the accuracies and losses.
        mean_val_accuracy = np.mean(val_batch_accuracies)
        mean_val_cost = np.mean(val_batch_costs)

        # Create a new Summary object with mean_val accuracy
        # and mean_val_loss and add it to Tensorboard.
        val_summary = tf.Summary(value=[
            tf.Summary.Value(tag="val_summaries/cost",
                             simple_value=mean_val_cost),
            tf.Summary.Value(tag="val_summaries/accuracy",
                             simple_value=mean_val_accuracy)])

        return mean_val_accuracy, mean_val_cost, val_summary

    def plot_train_stats(self):
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
        subplots[0, 0].plot(range(len(self.training_costs)), self.training_costs, color="blue")
        subplots[0, 0].set_title("Training cost")
        subplots[0, 0].grid()

        # Training accuracy plot
        subplots[0, 1].plot(range(len(self.validation_costs)), self.validation_costs, color="red")
        subplots[0, 1].set_title("Validation cost")
        subplots[0, 1].grid()

        # Validation cost plot
        subplots[1, 0].plot(range(len(self.training_accuracies)), self.training_accuracies, color="blue")
        subplots[1, 0].set_title("Training accuracy")
        subplots[1, 0].grid()

        # Validation accuracy plot
        subplots[1, 1].plot(range(len(self.validation_accuracies)), self.validation_accuracies, color="red")
        subplots[1, 1].set_title("Validation accuracy")
        subplots[1, 1].grid()

        # Space between plots
        f.subplots_adjust(hspace=0.3)

        plt.show()
