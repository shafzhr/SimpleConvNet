"""
Initializers
"""
import numpy as np


def he_normal(shape):
    """
    :param shape: tuple with the shape of the wanted output (filters_amount, depth, height, width)
    :return: array (it's shape=param shape) with initialized values using 'he normal' initializer
    """
    fan_in, _ = _calc_fans(shape)
    scale = 2 / fan_in
    # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
    stddev = np.sqrt(scale) / .87962566103423978
    return np.random.normal(loc=0, scale=stddev, size=shape)


def glorot_uniform(shape):
    """
    :param shape: tuple with the shape of the wanted output (filters_amount, depth, height, width)
    :return: array (it's shape=param shape) with initialized values using 'glorot uniform' initializer
    """
    fan_in, fan_out = _calc_fans(shape)
    scale = 1. / ((fan_in + fan_out) / 2.)
    limit = np.sqrt(3.0 * scale)
    return np.random.uniform(low=-limit, high=limit, size=shape)


def zeros(shape):
    """
    :param shape: tuple with the shape of the wanted output (filters_amount, depth, height, width)
    :return: array (it's shape=param shape) with initialized values using 'zeros' initializer
    """
    return np.zeros(shape=shape)


def ones(shape):
    """
    :param shape: tuple with the shape of the wanted output (filters_amount, depth, height, width)
    :return: array (it's shape=param shape) with initialized values using 'ones' initializer
    """
    return np.ones(shape=shape)


def _calc_fans(shape):
    """
    :param shape: tuple with the shape(4D - for example, filters, depth, width, height)
    :return: (fan_in, fan_out)
    """
    if len(shape) == 2:
        # Fully connected layer (units, input)
        fan_in = shape[1]
        fan_out = shape[0]

    elif len(shape) in {3, 4, 5}:
        # Convolutional kernals
        k_size = np.prod(shape[2:])
        fan_in = k_size * shape[1]
        fan_out = k_size * shape[0]

    else:
        raise ValueError("Incompatible shape")

    return fan_in, fan_out


WEIGHT_FUNCTIONS = {"he_normal": he_normal,
                    "glorot_uniform": glorot_uniform}
BIAS_FUNCTIONS = {"zeros": zeros,
                  "ones": ones}
