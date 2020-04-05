"""
Initializers
"""
import numpy as np


def he_normal(shape):
    """
    :param shape: tuple with the shape of the wanted output (filters_amount, depth, height, width)
    :return: array (it's shape=param shape) with initialized values using 'he normal' initializer
    """
    fan_in = np.prod(shape[1:])
    scale = 2 / fan_in
    # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
    stddev = np.sqrt(scale) / .87962566103423978
    return np.random.normal(loc=0, scale=stddev, size=shape)


def glorot_uniform(shape):
    """
    :param shape: tuple with the shape of the wanted output (filters_amount, depth, height, width)
    :return: array (it's shape=param shape) with initialized values using 'glorot uniform' initializer
    """
    pass


def zeros(shape):
    """
    :param shape: tuple with the shape of the wanted output (filters_amount, depth, height, width)
    :return: array (it's shape=param shape) with initialized values using 'zeros' initializer
    """
    return np.ones(shape=shape)


def ones(shape):
    """
    :param shape: tuple with the shape of the wanted output (filters_amount, depth, height, width)
    :return: array (it's shape=param shape) with initialized values using 'ones' initializer
    """
    return np.ones(shape=shape)


WEIGHT_FUNCTIONS = {"he_normal": lambda shape: he_normal(shape),
                    "glorot_uniform": lambda shape: glorot_uniform(shape)}
BIAS_FUNCTIONS = {"zeros": lambda shape: zeros(shape),
                  "ones": lambda shape: ones(shape)}
