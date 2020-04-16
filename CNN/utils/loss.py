"""
Loss functions
"""
import numpy as np


class CategoricalCrossEntropy:
    """
    Categorical cross entropy
    """
    @staticmethod
    def calc_loss(output, target):
        """
        Categorical cross entropy function
        :param output: ndarray of the output
        :param target: ndarray of the desired output
        :return: the loss (scalar)
        """
        output /= output.sum(axis=-1, keepdims=True)
        output = np.clip(output, 1e-7, 1 - 1e-7)
        return np.sum(target * -np.log(output), axis=-1, keepdims=False)

    @staticmethod
    def calc_derivative(output, target):
        """
        :param output: the output from the network
        :param target: training labels
        :return: the derivative of the loss functions
        """
        return target - output
