import numpy as np
from .layers.conv import Conv2D
from .utils.data import get_batches


def unit_test():
    """
    Unit testing
    """
    conv_test()
    data_test()


def conv_test():
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


def data_test():
    x = np.array([[[1, 2], [3, 4], [5, 6]],
                  [[11, 12], [13, 14], [15, 16]],
                  [[21, 22], [23, 24], [25, 26]],
                  [[31, 32], [33, 34], [35, 36]],
                  [[41, 42], [43, 44], [45, 46]],
                  [[51, 52], [53, 54], [55, 56]],
                  [[61, 62], [63, 64], [65, 66]]])
    y = np.array([[0], [1], [2], [3], [4], [5], [6]])

    print(list(get_batches(x, y, 3)))
    assert len(list(get_batches(x, y, 3))) == 3
