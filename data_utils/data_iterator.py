import numpy as np


class DataIterator(object):

    def __init__(self, data: np.array, labels: np.array, batch_size: int):

        if data.shape[0] != labels.shape[0]:
            raise ValueError("The number of data instances in the arrays doesn't match. "
                             "Found data_instances: {} and labels_instances: {}".format(data.shape[0],
                                                                                        labels.shape[0]))

        self.data = data
        self.labels = labels
        self.batch_size = batch_size

        self.data_instances = data.shape[0]
        self.full_batches = self.data_instances // self.batch_size

        if self.full_batches * self.batch_size == self.data_instances:
            self.total_batches = self.full_batches
        else:
            self.total_batches = self.full_batches + 1

    def __iter__(self):

        s = np.arange(self.data_instances)
        np.random.shuffle(s)

        self.data = self.data[s]
        self.labels = self.labels[s]

        for i in range(self.full_batches):
            data_batch = self.data[i * self.batch_size:(i+1) * self.batch_size, :]
            labels_batch = self.labels[i * self.batch_size:(i+1) * self.batch_size, :]
            yield i, data_batch, labels_batch

        if self.full_batches * self.batch_size != self.data_instances:
            data_batch = self.data[self.full_batches * self.batch_size:, :]
            labels_batch = self.labels[self.full_batches * self.batch_size:, :]
            yield self.total_batches - 1, data_batch, labels_batch


# Test code
test_data = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
test_labels = np.array([["a"], ["b"], ["c"], ["d"]])
print(test_data)
iterator = DataIterator(test_data, test_labels, 3)

for j in range(2):
    for batch in iterator:
        print(batch)
