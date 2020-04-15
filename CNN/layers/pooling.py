"""
Pooling layers
"""
from typing import (
    Tuple,
)
import numpy as np
from .layers import Layer


class MaxPooling2D(Layer):
    """
    Max-pooling layer
    """

    def __init__(self, pool_size: Tuple, stride: int):
        """
        :param pool_size: size of pooling window size(2x2 Tuple)
        :param stride: size of the stride
        """
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}

    def run(self, x):
        """
        Applying MaxPooling on `x`
        :param x: input
        :return: output of the MaxPooling on `x`
        """
        dim_x, h_x, w_x = x.shape
        h_poolwindow, w_poolwindow = self.pool_size

        out_h = int((h_x - h_poolwindow) / self.stride) + 1
        out_w = int((w_x - w_poolwindow) / self.stride) + 1

        out = np.zeros((dim_x, out_h, out_w))
        for ch in range(dim_x):
            curr_y = y_out = 0
            while curr_y + h_poolwindow <= h_x:
                curr_x = x_out = 0
                while curr_x + w_poolwindow <= w_x:
                    out[ch, y_out, x_out] = np.max(x[ch,
                                                   curr_y:curr_y + h_poolwindow,
                                                   curr_x:curr_x + w_poolwindow])
                    curr_x += self.stride
                    x_out += 1
                curr_y += self.stride
                y_out += 1

        self.cache['X'] = x
        return out

    def backprop(self, dA_prev):
        """Back propagation in a max pooling layer"""
        x = self.cache['X']
        dim_x, h_x, w_x = x.shape
        h_poolwindow, w_poolwindow = self.pool_size

        dA = np.zeros(shape=x.shape)  # dC/dA --> gradient of the input
        for ch in range(dim_x):
            curr_y = out_y = 0
            while curr_y + h_poolwindow <= h_x:
                curr_x = out_x = 0
                while curr_x + w_poolwindow <= w_x:
                    window_slice = x[ch, curr_y:curr_y + h_poolwindow, curr_x + w_poolwindow]
                    i, j = np.unravel_index(np.argmax(window_slice), window_slice.shape)
                    dA[ch, curr_y + i, curr_x + j] = dA_prev[ch, out_y, out_x]

                    curr_x += self.stride
                    out_x += 1

                curr_y += self.stride
                out_y += 1
        return dA
