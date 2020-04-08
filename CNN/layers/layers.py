"""
Layers
"""
import abc
import numpy as np


class Layer(metaclass=abc.ABCMeta):
    """Layer abstract class"""

    @abc.abstractmethod
    def run(self, x):
        """feed an input 'x' through the layer"""
        pass


class Dropout(Layer):
    """
    Dropout layer
    """

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

    def run(self, x):
        """
        Flattening array
        :param x: input to flatten
        """
        return x.flatten()


a = np.array([[1, 2], [3, 4]])
flatter = Flattening()
print(flatter.run(a))
