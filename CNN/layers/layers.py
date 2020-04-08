"""
Layers
"""
import abc


class Layer(metaclass=abc.ABCMeta):
    """Layer interface"""

    @abc.abstractmethod
    def run(self, x):
        """feed an input 'x' through the layer"""
        pass

    @abc.abstractmethod
    def initialize_weights(self, shape):
        """
        Initializing weights
        :param shape: wanted output's shape
        """
        pass

    @abc.abstractmethod
    def initialize_bias(self, shape):
        """
        Initializing biases
        :param shape: wanted output's shape
        """
        pass

    @abc.abstractmethod
    def activation(self, x):
        """
        use the layer's activation function over input 'x'
        :param x: input
        """
        pass

