"""
Pooling layers
"""
from typing import (
    Tuple,
)
import numpy as np
import skimage.measure
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

        maxs = out.repeat(2, axis=2).repeat(2, axis=3)
        x_window = x[:, :, :out_h * self.stride, :out_w * self.stride]
        mask = np.equal(x_window, maxs).astype(int)

        if is_training:
            self.cache['X'] = x
            self.cache['mask'] = mask
        return out

    def backprop(self, dA_prev):
        """
        Back propagation in a max pooling layer
        :param dA_prev: derivative of the cost function with respect to the previous layer(when going backwards)
        :return: the derivative of the cost layer with respect to the current layer
        """
        x = self.cache['X']
        h_poolwindow, w_poolwindow = self.pool_size

        mask = self.cache['mask']
        dA = dA_prev.repeat(h_poolwindow, axis=2).repeat(w_poolwindow, axis=3)
        dA = np.multiply(dA, mask)
        pad = np.zeros(x.shape)
        pad[:, :, :dA.shape[2], :dA.shape[3]] = dA
        return pad
