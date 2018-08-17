import logging
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np


logger = logging.getLogger()


DATA_FILE = "data/mushrooms.csv"


def data_preparation(data_split: float=0.1):

    # Read csv data to pandas dataframe
    logger.info("Reading data from '{}'".format(DATA_FILE))
    data = pd.read_csv(DATA_FILE)

    ##########################################################################
    #### Create labels array and remove it from the data variable        #####
    ##########################################################################

    logger.info("Generating labels array.")

    # Create a list with all the data points labels
    labels_data = list(data["class"])

    # Convert to numbers
    labels = []
    for label in labels_data:
        if label == "p":
            labels.append([0])
        elif label == "e":
            labels.append([1])

    labels = np.array(labels)
    logger.info("Labels matrix shape = {}".format(labels.shape))

    # Delete "class" column from the data
    data = data.drop(labels=["class"], axis=1)

    ##########################################################################
    #### One hot encode features from the dataset                        #####
    ##########################################################################

    logger.info("Converting data features to one hot encoded vectors.")

    data_matrix = []

    for column in data.columns:
        column_data = []
        column_values = list(data[column])

        counter = Counter(column_values)
        n_column_labels = len(counter)
        column_labels = list(counter.keys())

        for i in range(n_column_labels):
            column_data.append(np.zeros(len(column_values)))

        for j in range(len(column_values)):
            label_index = column_labels.index(column_values[j])
            column_data[label_index][j] = 1

        for data_column in column_data:
            data_matrix.append(data_column)

    data_matrix = np.array(data_matrix).transpose()

    logger.info("Final data matrix shape = {}".format(data_matrix.shape))

    ##########################################################################
    #### Split data into training, validation and test sets              #####
    ##########################################################################

    logger.info("Splitting data into training ({}%), validation ({}%) and "
                "test ({}%) sets.".format(100 - data_split * 2 * 100, data_split * 100, data_split * 100))

    split_instances = round(data_matrix.shape[0] * data_split)
    training_instances = data_matrix.shape[0] - split_instances * 2
    logger.info("Training instances: {}".format(training_instances))
    logger.info("Validation instances: {}".format(split_instances))
    logger.info("Test instances: {}".format(split_instances))

    training_data = data_matrix[:training_instances, :]
    training_labels = labels[:training_instances, :]
    logger.info("Training data shape = {}".format(training_data.shape))

    validation_data = data_matrix[training_instances:training_instances + split_instances, :]
    validation_labels = labels[training_instances:training_instances + split_instances, :]
    logger.info("Validation data shape = {}".format(validation_data.shape))

    test_data = data_matrix[training_instances + split_instances:training_instances + split_instances * 2, :]
    test_labels = labels[training_instances + split_instances:training_instances + split_instances * 2, :]
    logger.info("Testing data shape = {}".format(test_data.shape))

    ##########################################################################
    #### Save data sets to .npy files                                    #####
    ##########################################################################

    training_data_dir = Path("training_data")
    training_data_dir.mkdir(exist_ok=True)
    logger.info("Saving data sets to '{}'".format(training_data_dir))

    np.save(str(Path(training_data_dir, "mushrooms_training_data.npy")), training_data)
    np.save(str(Path(training_data_dir, "mushrooms_training_labels.npy")), training_labels)
    np.save(str(Path(training_data_dir, "mushrooms_validation_data.npy")), validation_data)
    np.save(str(Path(training_data_dir, "mushrooms_validation_labels.npy")), validation_labels)
    np.save(str(Path(training_data_dir, "mushrooms_test_data.npy")), test_data)
    np.save(str(Path(training_data_dir, "mushrooms_test_labels.npy")), test_labels)

    ##########################################################################
    #### Return generated files                                          #####
    ##########################################################################

    training = (training_data, training_labels)
    validation = (validation_data, validation_labels)
    test = (test_data, test_labels)

    return training, validation, test


def main():
    parser = argparse.ArgumentParser(description="Process mushrooms data to train a neural network.")

    # Optional arguments
    parser.add_argument("--split", type=float, default=0.1,
                        help="The portion of the data to use as validation and test sets. "
                             "Defaults to 0.1.")
    args = parser.parse_args()

    # Logging operations
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(filename)s - [ %(message)s ]',
                                  '%m/%d/%Y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    data_preparation(data_split=args.split)


if __name__ == '__main__':
    main()
