import numpy as np
import abc


class Activation(metaclass=abc.ABCMeta):
    """
    Activation abstract class
    """

    @abc.abstractmethod
    def apply(self, x):
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

    def apply(self, x):
        """
        Applying ReLU over `x`
        :param x: input (numpy array)
        """
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

    def apply(self, x):
        """
        Applying Softmax over `x`
        :param x: input (numpy array)
        """
        self.X = x.copy()
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / np.sum(exps, axis=0, keepdims=True)

    def backprop(self, dA_prev):
        """
        Back Propagation in Softmax
        :param dA_prev: derivative of the cost function with respect to the previous layer(when going backwards)
        :return: the derivative of the cost layer with respect to the current layer
        """
        return dA_prev * (self.X * (1 - self.X))


ACTIVATION_FUNCTIONS = {'relu': ReLU}
