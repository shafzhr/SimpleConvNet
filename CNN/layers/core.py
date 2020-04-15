"""
Core layers
"""
from .layers import Layer
import numpy as np


class Dropout(Layer):
    """
    Dropout layer
    """

    def __init__(self, rate: float):
        """
        :param rate: drop rate
        """
        if not 0 < rate < 1:
            raise ValueError("rate has to be between 0 and 1")

        self.rate = rate
        self.noise = None

    def run(self, x):
        """
        Applies dropout on `x`
        :param x: input array
        """
        shape = x.shape
        noise = np.random.choice([0, 1], shape, replace=True, p=[self.rate, 1 - self.rate])
        self.noise = noise
        return x * noise / (1 - self.rate)

    def backprop(self, dA_prev):
        """
        Back propagation in a flattening layer
        :param dA_prev: derivative of the cost function with respect to the previous layer(when going backwards)
        :return: the derivative of the cost layer with respect to the current layer
        """
        dA = dA_prev * self.noise
        return dA / (1 - self.rate)


class Flattening(Layer):
    """
    Flattening layer
    """

    def __init__(self):
        self.shape = ()

    def run(self, x):
        """
        Flattening array
        :param x: input to flatten
        """
        self.shape = x.shape
        return x.reshape((np.prod(x.shape), 1))

    def backprop(self, dA_prev):
        """
        Back propagation in a flattening layer
        Reshapes the input to the same shape as the previous layer's output
        :param dA_prev: derivative of the cost function with respect to the previous layer(when going backwards)
        :return: the derivative of the cost layer with respect to the current layer
        """
        return dA_prev.reshape(self.shape)
