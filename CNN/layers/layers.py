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
