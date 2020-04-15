"""
Convolutional layers
"""
from typing import (
    Tuple,
)
from .layers import Layer, Trainable
import numpy as np
from ..utils.initializers import WEIGHT_FUNCTIONS, BIAS_FUNCTIONS
from ..utils.activations import ACTIVATION_FUNCTIONS


class ConvLayer(Layer, Trainable):
    """Convolutional layer"""

    def __init__(self, filters_amount: int, filter_size: Tuple[int], activation: str, filter_initializer: str,
                 bias_initializer: int, stride: int, input_d: int, **kw):
        """
        :param filters_amount: layer's amount of filters
        :param filter_size: the size of the kernal (height, width)
        :param activation: the used activations function
        :param filter_initializer: the used filter initializer function
        :param bias_initializer: the used bias initializer function
        :param stride: the stride of the convolution
        :param input_d: input's dimension
        """
        if len(filter_size) != 2:
            raise ValueError("Filter size has to be a Tuple of 2 elements(height, width)")
        self.filter_size = filter_size
        self.activation_name = activation
        self.stride = stride
        self.filter_initializer = filter_initializer
        self.filters = self.initialize_weights((filters_amount, input_d, *filter_size))
        self.bias_initializer = bias_initializer
        self.bias = self.initialize_bias((filters_amount, 1))
        self.cache = {}
        self.grads = {}

    def initialize_weights(self, shape):
        """
        Initializing filters
        :param shape: Tuple, wanted output's shape(filters_amount, depth, width, height)
        :return: numpy array of weights(it's shape= the given shape)
        """
        if self.filter_initializer not in WEIGHT_FUNCTIONS.keys():
            raise ValueError("Filter initializer name has to be in {}".format(str(WEIGHT_FUNCTIONS.keys())))

        return WEIGHT_FUNCTIONS[self.filter_initializer](shape)

    def initialize_bias(self, shape):
        """
        Initializing biases
        :param shape: Tuple, wanted output's shape(biases_amount, depth, width, height)
        """
        if self.bias_initializer not in BIAS_FUNCTIONS.keys():
            raise ValueError("Bias initializer name has to be in {}".format(str(BIAS_FUNCTIONS.keys())))

        return BIAS_FUNCTIONS[self.bias_initializer](shape)

    def activation(self, x):
        """
        use the layer's activation function over input 'x'
        :param x: input
        :return nothing as the input array is passed by reference
        """
        if self.activation_name not in ACTIVATION_FUNCTIONS.keys():
            raise ValueError("Activation name has to be in {}".format(str(ACTIVATION_FUNCTIONS.keys())))

        ACTIVATION_FUNCTIONS[self.activation_name](x)

    def run(self, x):
        """Convolves the filters over 'x' """
        self.cache['X'] = x

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

    def backprop(self, dA_prev):
        """
        Back Propagation in a convolutional layer.
        calculating this layer's dA using the chain rule.
        For more info:
        https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e and:
        https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199

        :param dA: previous layer's derivative.
        :return: Current layer's derivative
        """
        x = self.cache['x']

        n_filt, dim_filt, size_filt, _ = self.filters.shape
        _, size_img, _ = x.shape

        dA = np.zeros(shape=x.shape)  # dC/dA --> gradient of the input
        dF = np.zeros(shape=self.filters.shape)  # dC/dF --> gradient of the filters
        dB = np.zeros(shape=self.bias.shape)  # dC/dB --> gradient of the biases

        for filt in range(n_filt):
            y_filt = y_out = 0
            while y_filt + size_filt <= size_img:
                x_filt = x_out = 0
                while x_filt + size_filt <= size_img:
                    dF[filt] += dA_prev[filt, y_out, x_out] * x[:, y_filt:y_filt + size_filt, x_filt:x_filt + size_filt]

                    dA[:, y_filt:y_filt + size_filt, x_filt:x_filt + size_filt] += (
                                dA_prev[filt, y_out, x_out] * self.filters[filt])

                    x_filt += self.stride
                    x_out += 1

                y_filt += self.stride
                x_out += 1
            dB += np.sum(dA_prev[filt])

        self.grads['dF'] = dF
        self.grads['dB'] = dB
        return dA

    def update_params(self):
        pass


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
