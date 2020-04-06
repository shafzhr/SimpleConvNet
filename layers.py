"""
Layers
"""
from typing import (
    Tuple,
)
import abc
import numpy as np
from utils.initializers import WEIGHT_FUNCTIONS, BIAS_FUNCTIONS


class Layer(metaclass=abc.ABCMeta):
    """Layer interface"""

    @abc.abstractmethod
    def run(self, x):
        """feed an input 'x' through the layer"""
        raise NotImplementedError('users must define "run" to use this base class')


class ConvLayer(Layer):
    """Convolutional layer"""

    def __init__(self, filters_amount: int, filter_size: Tuple[int], activation: str, filter_initializer: str,
                 bias_initializer: int, stride: int, input_d: int, **kw):
        """
        :param filters_amount: layer's amount of filters
        :param filter_size: the size of the kernal
        :param activation: the used activations function
        :param filter_initializer: the used filter initializer function
        :param bias_initializer: the used bias initializer function
        :param stride: the stride of the convolution
        :param input_d: input's dimension
        """
        self.filter_size = filter_size
        self.activation = activation
        self.stride = stride
        self.filter_initializer = filter_initializer
        self.filters = self.initialize_filters((filters_amount, input_d, *filter_size))
        self.bias_initializer = bias_initializer
        self.bias = self.initialize_bias((filters_amount, input_d, *filter_size))

    def initialize_filters(self, shape):
        """
        Initializing filters
        :param shape:
        :return:
        """
        if self.filter_initializer not in WEIGHT_FUNCTIONS.keys():
            raise ValueError("Filter initializer name has to be in {}".format(str(WEIGHT_FUNCTIONS.keys())))

        return WEIGHT_FUNCTIONS[self.filter_initializer](shape)

    def initialize_bias(self, shpae):
        """
        Initializing biases
        """
        if self.bias_initializer not in BIAS_FUNCTIONS.keys():
            raise ValueError("Bias initializer name has to be in {}".format(str(BIAS_FUNCTIONS.keys())))

        return BIAS_FUNCTIONS[self.bias_initializer](shpae)

    def run(self, x):
        """Convolves the filters over 'x' """
        n_filt, dim_filt, size_filt, _ = self.filters.shape
        dim_img, size_img, _ = x.shape

        if dim_filt != dim_img:
            raise ValueError("Image and filter dimension must be the same")

        size_out = int((size_img - size_filt) / self.stride) + 1

        out = np.zeros((n_filt, size_out, size_out))
        for filt in range(n_filt):
            y_filt = y_out = 0
            while y_filt + size_filt <= size_img:
                x_filt = x_out = 0
                while x_filt + size_filt <= size_img:
                    out[filt, y_out, x_out] = np.sum(
                        self.filters[filt] * (x[:, y_filt: y_filt + size_filt, x_filt:x_filt + size_filt])
                        + self.bias[filt]
                    )
                    x_filt += self.stride
                    x_out += 1
                y_filt += self.stride
                y_out += 1

        return out


class Conv2D(ConvLayer):
    """2D Convolutional layer"""

    def __init__(self, filters_amount: int, filter_size: Tuple[int], activation: str, filter_initializer: str,
                 bias_initializer: int, stride: int):
        """
        :param filters_amount: layer's amount of filters
        :param filter_size: the size of the kernal
        :param activation: the used activations function
        :param filter_initializer: the used filter initializer function
        :param bias_initializer: the used bias initializer function
        :param stride: the stride of the convolution
        """
        super().__init__(filters_amount=filters_amount,
                         filter_size=filter_size,
                         activation=activation,
                         stride=stride,
                         filter_initializer=filter_initializer,
                         bias_initializer=bias_initializer,
                         input_d=2
                         )

    def run(self, x):
        """Convolves the filters over 'x' """
        return super().run(x)
