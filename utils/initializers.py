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
    stddev = np.sqrt(scale) / .87962566103423978
    return np.random.normal(loc=0, scale=stddev, size=shape)
