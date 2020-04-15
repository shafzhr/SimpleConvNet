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

    def run(self, x):
        """
        Applies dropout on `x`
        :param x: input array
        """
        shape = x.shape
        noise = np.random.choice([0, 1], shape, replace=True, p=[self.rate, 1 - self.rate])
        return x * noise / (1 - self.rate)

    def backprop(self, dA):
        pass


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
        :param dA_prev: derivative of the previous layer(when going backwards)
        :return:
        """
        return dA_prev.reshape(self.shape)
