"""
This script has classes to iterate over data and produce batches of it,
that can be used to train neural network models.
"""

import numpy as np


class DataIterator(object):
    """
    Class used to iterate over a dataset that is already converted to numpy arrays. One array
    should have the input data, and another should have the target data (the expected output).
    This class allows to iterate the data in batches of any length.
    """

    def __init__(self, data: np.array, labels: np.array, batch_size: int, shuffle: bool):
        """
        Create a new data iterator, for a specific dataset. The batches are constructed
        by splitting the data and the labels on the first dimension.

        :param data: An array with the input data.
        :param labels: An array with the true labels for the input data.
        :param batch_size: The size of the generated batches.
        :param shuffle: Activate to shuffle the data before generating the batches.
        """

        if data.shape[0] != labels.shape[0]:
            raise ValueError("The number of data instances in the arrays doesn't match. "
                             "Found {} data_instances and {} labels_instances".format(data.shape[0],
                                                                                      labels.shape[0]))

        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data_instances = data.shape[0]
        self.full_batches = self.data_instances // self.batch_size

        if self.full_batches * self.batch_size == self.data_instances:
            self.total_batches = self.full_batches
        else:
            self.total_batches = self.full_batches + 1

    def __iter__(self) -> (np.array, np.array):
        """
        Iterate one time over the entire data, generating batches.
        This will generate "self.total_batches" number of batches.

        :return: A tuple with the data batch and the labels batch.
        """

        # Shuffle the data before generating the batches, if indicated.
        if self.shuffle:
            s = np.arange(self.data_instances)
            np.random.shuffle(s)

            self.data = self.data[s]
            self.labels = self.labels[s]

        # Generate full size batches
        for i in range(self.full_batches):
            data_batch = self.data[i * self.batch_size:(i+1) * self.batch_size, :]
            labels_batch = self.labels[i * self.batch_size:(i+1) * self.batch_size, :]
            yield data_batch, labels_batch

        # Generate last batch with the remaining data
        if self.full_batches * self.batch_size != self.data_instances:
            data_batch = self.data[self.full_batches * self.batch_size:, :]
            labels_batch = self.labels[self.full_batches * self.batch_size:, :]
            yield data_batch, labels_batch


# # Test code. Uncomment to test this script.
# test_data = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
# test_labels = np.array([[0], [1], [2], [3]])
# print("Test data:\n{}".format(test_data))
# print("Test labels:\n{}".format(test_labels))
#
# iterator = DataIterator(data=test_data,
#                         labels=test_labels,
#                         batch_size=3,
#                         shuffle=True)
#
# for j in range(2):
#     for batch in iterator:
#         print(batch)
