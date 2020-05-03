import numpy as np
from .layers.conv import Conv2D


def unit_test():
    """
    Unit testing
    """
    input_matrix = np.array([[[0, 1, 1, 1, 0, 0, 0],
                              [0, 0, 1, 1, 1, 0, 0],
                              [0, 0, 0, 1, 1, 1, 0],
                              [0, 0, 0, 1, 1, 0, 0],
                              [0, 0, 1, 1, 0, 0, 0],
                              [0, 1, 1, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 0]]])
    input_copy = input_matrix.copy()
    kernal = np.array([[[[1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]]]])
    expected = np.array([[[1, 4, 3, 4, 1],
                          [1, 2, 4, 3, 3],
                          [1, 2, 3, 4, 1],
                          [1, 3, 3, 1, 1],
                          [3, 3, 1, 1, 0]]])

    conv_layer = Conv2D(filters_amount=1,
                        filter_size=kernal.shape[2:],
                        activation='relu',
                        filter_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        stride=1
                        )
    conv_layer.filters = kernal

    output = conv_layer.run(input_matrix)
    assert np.allclose(output, expected)
    assert (input_matrix == input_copy).all()

