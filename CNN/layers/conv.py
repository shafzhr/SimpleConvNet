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
            self.filters = self.initialize_weights((self.units, x.shape[1], *self.filter_size))
            self.grads = self._init_bias_weight_like()

        if is_training:
            self.cache['X'] = x

        n_filt, ch_filt, h_filt, w_filt = self.filters.shape
        n_batch, ch, h, w = x.shape

        if ch_filt != ch:
            raise ValueError("Image and filter dimension must be the same")

        h_out = (h - h_filt) // self.stride + 1
        w_out = (w - w_filt) // self.stride + 1

        sub_windows = np.lib.stride_tricks.as_strided(x, (n_batch, h_out, w_out, ch, h_filt, w_filt),
                                                      (x.strides[0], x.strides[2] * self.stride,
                                                       x.strides[3] * self.stride, x.strides[1],
                                                       x.strides[2], x.strides[3])
                                                      )
        out = np.tensordot(sub_windows, self.filters, axes=[(3, 4, 5), (1, 2, 3)])
        out = out.transpose((0, 3, 1, 2))

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

        n_filt, ch_filt, h_filt, w_filt = self.filters.shape
        n_bathc, ch_img, h, w = x.shape
        _, dA_ch, dA_h, dA_w = dA_prev.shape

        dB = np.zeros(shape=self.bias.shape)  # dC/dB --> gradient of the biases
        dB += np.sum(dA_prev)

        as_strided = np.lib.stride_tricks.as_strided

        sub_windows = as_strided(x,
                                 shape=(n_bathc, ch_img, h_filt, w_filt, dA_h, dA_w),
                                 strides=(x.strides[0], x.strides[1],
                                          x.strides[2] * self.stride,
                                          x.strides[3] * self.stride,
                                          x.strides[2], x.strides[3])
                                 )
        F = np.tensordot(sub_windows, dA_prev, axes=[(0, -2, -1), (0, 2, 3)])
        dF = F.transpose((3, 0, 1, 2))

        pad_h = dA_h - 1
        pad_w = dA_w - 1
        pad_filt = np.pad(self.filters, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant')
        sub_windows = as_strided(pad_filt,
                                 shape=(n_filt, h, w, dA_h, dA_w, ch_filt),
                                 strides=(pad_filt.strides[0], pad_filt.strides[2],
                                          pad_filt.strides[3], pad_filt.strides[2],
                                          pad_filt.strides[3], pad_filt.strides[1])
                                 )

        dA = np.tensordot(sub_windows, dA_prev[:, :, ::-1, ::-1], axes=[(0, 3, 4), (1, 2, 3)])
        dA = dA.transpose((3, 2, 0, 1))

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
