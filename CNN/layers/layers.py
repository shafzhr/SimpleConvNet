"""
Layers
"""
import abc


class Layer(metaclass=abc.ABCMeta):
    """Layer abstract class"""

    @property
    @abc.abstractmethod
    def has_weights(self):
        """
        boolean, determines if the layer has weights
        """
        pass

    @property
    @abc.abstractmethod
    def has_bias(self):
        """
        boolean, determines if the layer has biases
        """
        pass

    @abc.abstractmethod
    def run(self, x):
        """
        feed an input 'x' through the layer
        """
        pass

    @abc.abstractmethod
    def backprop(self):
        """
        Back propagate
        """
        pass
