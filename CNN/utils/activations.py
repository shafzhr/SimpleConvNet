import numpy as np
import abc


class Activation(metaclass=abc.ABCMeta):
    """
    Activation abstract class
    """

    @abc.abstractmethod
    def apply(self, x, is_training):
        """
        Applying the activation function over `x`
        """
        pass

    @abc.abstractmethod
    def backprop(self, dA_prev):
        """
        Back Propagation in an activation function
        """
        pass


class ReLU(Activation):
    """
    ReLU activation function
    """

    def __init__(self):
        self.X = None

    def apply(self, x, is_training=True):
        """
        Applying ReLU over `x`
        :param x: input (numpy array)
        :param is_training: a boolean indicating whether training or not
        """
        if is_training:
            self.X = x.copy()
        x[x < 0] = 0
        return x

    def backprop(self, dA_prev):
        """
        Back Propagation in ReLU
        :param dA_prev: derivative of the cost function with respect to the previous layer(when going backwards)
        :return: the derivative of the cost layer with respect to the current layer
        """
        return dA_prev * np.where(self.X > 0, 1, 0)


class Softmax(Activation):
    """
    Softmax activation
    """

    def __init__(self):
        self.X = None

    def apply(self, x, is_training=True):
        """
        Applying Softmax over `x`
        :param is_training: a boolean indicating whether training or not
        :param x: input (numpy array)
        """
        if is_training:
            self.X = x.copy()
        shiftx = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shiftx)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        return probs

    def backprop(self, dA_prev):
        """
        Back Propagation in Softmax
        :param dA_prev: derivative of the cost function with respect to the previous layer(when going backwards)
        :return: the derivative of the cost layer with respect to the current layer
        """
        return dA_prev * (self.X * (1 - self.X))


ACTIVATION_FUNCTIONS = {'relu': ReLU, 'softmax': Softmax}
