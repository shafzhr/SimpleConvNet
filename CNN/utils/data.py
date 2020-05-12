import numpy as np


def get_batches(data, labels, batch_size):
    """
    :param data: numpy array with the data
    :param labels: numpy array with the labels corresponding to data
    :param batch_size: size of the batches
    :return: batch data and labels
    """
    assert len(data) == len(labels)
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], labels[i:i + batch_size]
