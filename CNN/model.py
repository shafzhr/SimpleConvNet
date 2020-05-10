import numpy as np
from typing import (
    List,
    Tuple
)
from CNN.layers.layers import Layer, Trainable
from CNN.utils.data import get_batches


class Model:
    """
    Model of neural network
    """

    def __init__(self, layers: List = None):
        """
        :param layers: model's layers
        """
        self.layers = list(layers)
        self.loss = None

    def add(self, layer):
        """
        Add a layer to the model
        :param layer: Instance of Layer
        """
        if not isinstance(layer, Layer):
            raise ValueError("Layer has to be an instance of Layer")
        self.layers.append(layer)

    def train(self, data: Tuple[List, List, List, List], batch_size: int, epochs: int, optimizer: str,
              **optimizer_params):
        """
        Train the model using a given optimizer
        :param data: a tuple of lists(x_train, y_train, x_test, y_test)
        :param batch_size: size of the batches
        :param epochs: amount of training epochs
        :param optimizer: optimizers name
        :param optimizer_params: configuration params for the optimizer function
        """
        if len(data) != 4:
            raise ValueError("data must be a Tuple of 4 lists")
        X_train, X_test, y_train, y_test = data
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

        iteration = 1
        for epoch in range(epochs):
            print("Epoch #{}".format(epoch + 1))
            for x_batch, y_batch in get_batches(X_train, y_train, batch_size):
                for x, y in zip(x_batch, y_batch):
                    x_preds = x.copy()
                    for layer in self.layers:
                        x_preds = layer.run(x_preds)
                    dA = self.loss.calc_loss(x_preds, y)
                    for layer in reversed(self.layers):
                        dA = layer.backprop(dA)
                for layer in self.layers:
                    if isinstance(layer, Trainable):
                        layer.update_params('adam', batch_size, **optimizer_params, t=iteration)
            iteration += batch_size

