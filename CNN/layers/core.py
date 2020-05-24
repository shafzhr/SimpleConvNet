"""
Core layers
"""
from .layers import Layer, Trainable
import numpy as np
from ..utils.initializers import WEIGHT_FUNCTIONS, BIAS_FUNCTIONS
from ..utils.activations import ACTIVATION_FUNCTIONS


class Dropout(Layer):
    """
    Dropout layer
    """

    def __init__(self, rate: float):
        """
        :param rate: drop rate
        """
        if not 0 < rate < 1:
            raise ValueError("rate has to be between 0 and 1")

        self.rate = rate
        self.noise = None

    def run(self, x, is_training=True):
        """
        Applies dropout on `x`
        :param is_training:
        :param x: input array
        """
        if not is_training:
            return x
        pKeep = 1 - self.rate
        weights = np.ones(x.shape)
        noise = np.random.rand(*weights.shape) < pKeep  # !!!
        self.noise = noise
        res = np.multiply(weights, noise)
        return res / pKeep

        # shape = x.shape
        # noise = np.random.choice([0, 1], shape, replace=True, p=[self.rate, 1 - self.rate])
        # self.noise = noise
        # return x * noise / (1 - self.rate)

    def backprop(self, dA_prev):
        """
        Back propagation in a flattening layer
        :param dA_prev: derivative of the cost function with respect to the previous layer(when going backwards)
        :return: the derivative of the cost layer with respect to the current layer
        """
        dA = dA_prev * self.noise
        return dA / (1 - self.rate)


class Flattening(Layer):
    """
    Flattening layer
    """

    def __init__(self):
        self.shape = ()

    def run(self, x, is_training=True):
        """
        Flattening array
        :param is_training:
        :param x: input to flatten
        """
        if is_training:
            self.shape = x.shape
        return x.flatten('K').reshape((x.shape[0], -1)).T

    def backprop(self, dA_prev):
        """
        Back propagation in a flattening layer
        Reshapes the input to the same shape as the previous layer's output
        :param dA_prev: derivative of the cost function with respect to the previous layer(when going backwards)
        :return: the derivative of the cost layer with respect to the current layer
        """
        return dA_prev.T.reshape(self.shape)


class Dense(Layer, Trainable):
    """Regular fully connected layer"""

    def __init__(self, units: int, activation: str, weight_initializer: str, bias_initializer: str):

        if activation not in ACTIVATION_FUNCTIONS.keys():
            raise ValueError("Activation name has to be in {}".format(str(ACTIVATION_FUNCTIONS.keys())))

        self.units = units
        self.activation = ACTIVATION_FUNCTIONS[activation]()
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.weights = None
        self.bias = self.initialize_bias((units, 1))
        self.cache = {}
        self.grads = {}

    def initialize_weights(self, shape):
        if self.weight_initializer not in WEIGHT_FUNCTIONS.keys():
            raise ValueError("Weight initializer name has to be in {}".format(str(WEIGHT_FUNCTIONS.keys())))

        return WEIGHT_FUNCTIONS[self.weight_initializer](shape)

    def initialize_bias(self, shape):
        if self.bias_initializer not in BIAS_FUNCTIONS.keys():
            raise ValueError("Bias initializer name has to be in {}".format(str(BIAS_FUNCTIONS.keys())))

        return BIAS_FUNCTIONS[self.bias_initializer](shape)

    def run(self, x, is_training=True):
        if self.weights is None:
            self.weights = self.initialize_weights((self.units, x.shape[0]))
            self.grads = self._init_bias_weight_like()

        if is_training:
            self.cache['X'] = x

        x = np.dot(self.weights, x) + self.bias
        return self.activation.apply(x, is_training)

    def backprop(self, dA_prev):
        """
        :param dA_prev: derivative of the cost function with respect to the previous layer(when going backwards)
        :return: the derivative of the cost function with respect to the current layer
        """
        dA_prev = self.activation.backprop(dA_prev)
        x = self.cache['X']
        self.grads['dW'] += np.dot(dA_prev, x.transpose())
        self.grads['dB'] += np.sum(dA_prev, axis=1).reshape((-1, 1))
        return np.dot(self.weights.transpose(), dA_prev)

    def _init_bias_weight_like(self):
        d = {'dW': np.zeros_like(self.weights), 'dB': np.zeros_like(self.bias)}
        return d

    def _calc_momentum(self, beta):
        """
        :param beta:
        :return:
        """
        if 'momentum' not in self.grads.keys():
            self.grads['momentum'] = self._init_bias_weight_like()
        self.grads['momentum']['dW'] = beta * self.grads['momentum']['dW'] + (1-beta) * self.grads['dW']
        self.grads['momentum']['dB'] = beta * self.grads['momentum']['dB'] + (1-beta) * self.grads['dB']

    def _calc_rmsprop(self, beta):
        """
        :param beta:
        :return:
        """
        if 'rmsprop' not in self.grads.keys():
            self.grads['rmsprop'] = self._init_bias_weight_like()
        self.grads['rmsprop']['dW'] = beta * self.grads['rmsprop']['dW'] + (1-beta) * np.power(self.grads['dW'], 2)
        self.grads['rmsprop']['dB'] = beta * self.grads['rmsprop']['dB'] + (1-beta) * np.power(self.grads['dB'], 2)

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

            self.weights -= kwarg['lr'] * w_m_hat / (np.sqrt(w_v_hat) + kwarg['epsilon'])
            self.bias -= kwarg['lr'] * b_m_hat / (np.sqrt(b_v_hat) + kwarg['epsilon'])

        self.grads['dW'].fill(0)
        self.grads['dB'].fill(0)
