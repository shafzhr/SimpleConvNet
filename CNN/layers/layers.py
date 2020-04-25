"""
Layers
"""
import abc


class Layer(metaclass=abc.ABCMeta):
    """Layer abstract class"""

    @abc.abstractmethod
    def run(self, x):
        """
        feed an input 'x' through the layer
        """
        pass

    @abc.abstractmethod
    def backprop(self, dA_prev):
        """
        Back propagate
        """
        pass


class Trainable(metaclass=abc.ABCMeta):
    """Trainable layer abstract class"""
    @abc.abstractmethod
    def update_params(self, optimizer, **kwargs):
        """
        updating params using gradients
        """
        pass
