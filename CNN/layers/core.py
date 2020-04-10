"""
Core layers
"""
from .layers import Layer
import numpy as np


class Dropout(Layer):
    """
    Dropout layer
    """

    @property
    def has_weights(self):
        return False

    @property
    def has_bias(self):
        return False

    def __init__(self, rate):
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


class Flattening(Layer):
    """
    Flattening layer
    """

    @property
    def has_weights(self):
        return False

    @property
    def has_bias(self):
        return False

    def run(self, x):
        """
        Flattening array
        :param x: input to flatten
        """
        return x.flatten()
