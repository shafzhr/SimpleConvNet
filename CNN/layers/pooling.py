"""
Pooling layers
"""
from typing import (
    Tuple,
)
import numpy as np
from numpy.lib.stride_tricks import as_strided
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

    def run(self, x, is_training=True):
        """
        Applying MaxPooling on `x`
        :param x: input
        :param is_training: a boolean indicating whether training or not
        :return: output of the MaxPooling on `x`
        """
        n_batch, ch_x, h_x, w_x = x.shape
        h_poolwindow, w_poolwindow = self.pool_size

        out_h = int((h_x - h_poolwindow) / self.stride) + 1
        out_w = int((w_x - w_poolwindow) / self.stride) + 1

        windows = as_strided(x,
                             shape=(n_batch, ch_x, out_h, out_w, *self.pool_size),
                             strides=(x.strides[0], x.strides[1],
                                      self.stride * x.strides[2],
                                      self.stride * x.strides[3],
                                      x.strides[2], x.strides[3])
                             )
        out = np.max(windows, axis=(4, 5))

        if is_training:
            self.cache['X'] = x
        return out

    def backprop(self, dA_prev):
        """
        Back propagation in a max pooling layer
        :param dA_prev: derivative of the cost function with respect to the previous layer(when going backwards)
        :return: the derivative of the cost layer with respect to the current layer
        """
        x = self.cache['X']
        n_batch, ch_x, h_x, w_x = x.shape
        h_poolwindow, w_poolwindow = self.pool_size

        dA = np.zeros(shape=x.shape)  # dC/dA --> gradient of the input
        for n in range(n_batch):
            for ch in range(ch_x):
                curr_y = out_y = 0
                while curr_y + h_poolwindow <= h_x:
                    curr_x = out_x = 0
                    while curr_x + w_poolwindow <= w_x:
                        window_slice = x[n, ch, curr_y:curr_y + h_poolwindow, curr_x:curr_x + w_poolwindow]
                        i, j = np.unravel_index(np.argmax(window_slice), window_slice.shape)
                        dA[n, ch, curr_y + i, curr_x + j] = dA_prev[n, ch, out_y, out_x]

                        curr_x += self.stride
                        out_x += 1

                    curr_y += self.stride
                    out_y += 1
        return dA
