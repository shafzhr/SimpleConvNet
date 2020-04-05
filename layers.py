"""
Layers
"""
from typing import (
    Tuple,
)
import abc
import numpy as np


class Layer(metaclass=abc.ABCMeta):
    """Layer interface"""

    @abc.abstractmethod
    def run(self, x):
        """feed an input 'x' through the layer"""
        raise NotImplementedError('users must define "run" to use this base class')
