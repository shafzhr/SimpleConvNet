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
                 bias_initializer: str, stride: int, **kw):
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

        if activation not in ACTIVATION_FUNCTIONS.keys():
            raise ValueError("Activation name has to be in {}".format(str(ACTIVATION_FUNCTIONS.keys())))

        self.units = filters_amount
        self.filter_size = filter_size
        self.activation = ACTIVATION_FUNCTIONS[activation]()
        self.stride = stride
        self.filter_initializer = filter_initializer
        self.filters = None
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

    def run(self, x, is_training=True):
        """Convolves the filters over 'x' """
        if self.filters is None:
            self.filters = self.initialize_weights((self.units, x.shape[0], *self.filter_size))
            self.grads = self._init_bias_weight_like()

        if is_training:
            self.cache['X'] = x

        n_filt, dim_filt, size_filt, _ = self.filters.shape
        dim_img, size_img, _ = x.shape

        if dim_filt != dim_img:
            raise ValueError("Image and filter dimension must be the same")

        size_out = int((size_img - size_filt) / self.stride) + 1

        ch, h, w = x.shape
        Hout = (h - self.filters.shape[-2]) // self.stride + 1
        Wout = (w - self.filters.shape[-1]) // self.stride + 1

        a = np.lib.stride_tricks.as_strided(x, (Hout, Wout, ch, self.filters.shape[2], self.filters.shape[3]),
                                            (x.strides[1] * self.stride, x.strides[2] * self.stride) + (
                                            x.strides[0], x.strides[1], x.strides[2]))
        out = np.einsum('ijckl,ackl->aij', a, self.filters)
        for b in self.bias:
            out += b

        out = self.activation.apply(out, is_training)
        return out

    def backprop(self, dA_prev):
        """
        Back Propagation in a convolutional layer.
        calculating this layer's dA using the chain rule.
        For more info:
        https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e and:
        https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199

        :param dA_prev: derivative of the cost function with respect to the previous layer(when going backwards)
        :return: the derivative of the cost layer with respect to the current layer
        """
        dA_prev = self.activation.backprop(dA_prev)

        x = self.cache['X']

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

        self.grads['dW'] += dF
        self.grads['dB'] += dB
        return dA

    def _init_bias_weight_like(self):
        d = {'dW': np.zeros_like(self.filters), 'dB': np.zeros_like(self.bias)}
        return d

    def _calc_momentum(self, beta):
        """
        :param beta:
        :return:
        """
        if 'momentum' not in self.grads:
            self.grads['momentum'] = self._init_bias_weight_like()
        self.grads['momentum']['dW'] = beta * self.grads['momentum']['dW'] + (1 - beta) * self.grads['dW']
        self.grads['momentum']['dB'] = beta * self.grads['momentum']['dB'] + (1 - beta) * self.grads['dB']

    def _calc_rmsprop(self, beta):
        """
        :param beta:
        :return:
        """
        if 'rmsprop' not in self.grads:
            self.grads['rmsprop'] = self._init_bias_weight_like()
        self.grads['rmsprop']['dW'] = beta * self.grads['rmsprop']['dW'] + (1 - beta) * np.power(self.grads['dW'], 2)
        self.grads['rmsprop']['dB'] = beta * self.grads['rmsprop']['dB'] + (1 - beta) * np.power(self.grads['dB'], 2)

    def update_params(self, optimizer, batch_size, **kwarg):
        """
        :param optimizer:
        :param batch_size:
        :param kwarg:
        :return:
        """
        self.grads['dW'] /= batch_size
        self.grads['dB'] /= batch_size

        if optimizer == 'adam':
            if not all([arg in {'lr', 'beta1', 'beta2', 't', 'epsilon'} for arg in kwarg]):
                raise ValueError("Incorrect arguments")

            self._calc_momentum(kwarg['beta1'])
            self._calc_rmsprop(kwarg['beta2'])

            w_m_hat = self.grads['momentum']['dW'] / (1 - np.power(kwarg['beta1'], kwarg['t']))
            b_m_hat = self.grads['momentum']['dB'] / (1 - np.power(kwarg['beta1'], kwarg['t']))
            w_v_hat = self.grads['rmsprop']['dW'] / (1 - np.power(kwarg['beta2'], kwarg['t']))
            b_v_hat = self.grads['rmsprop']['dB'] / (1 - np.power(kwarg['beta2'], kwarg['t']))

            self.filters -= kwarg['lr'] * w_m_hat / (np.sqrt(w_v_hat) + kwarg['epsilon'])
            self.bias -= kwarg['lr'] * b_m_hat / (np.sqrt(b_v_hat) + kwarg['epsilon'])

        self.grads['dW'].fill(0)
        self.grads['dB'].fill(0)


class Conv2D(ConvLayer):
    """2D Convolutional layer"""

    def __init__(self, filters_amount: int, filter_size: Tuple[int], activation: str, filter_initializer: str,
                 bias_initializer: str, stride: int):
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
                         )
