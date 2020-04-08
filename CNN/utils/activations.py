import numpy as np


def relu(x):
    """
    puts relu over input 'x'
    :param x: input(numpy array)
    """
    x[x < 0] = 0


ACTIVATION_FUNCTIONS = {'relu': relu}
