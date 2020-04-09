"""
Loss functions
"""
import numpy as np


def categorical_crossentropy(output, target):
    """
    Categorical cross entropy function
    :param output: ndarray of the output
    :param target: ndarray of the desired output
    :return: the loss (scalar)
    """
    output /= output.sum(axis=-1, keepdims=True)
    output = np.clip(output, 1e-7, 1 - 1e-7)
    return np.sum(target * -np.log(output), axis=-1, keepdims=False)
