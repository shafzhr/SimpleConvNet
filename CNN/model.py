import numpy as np
from typing import (
    List,
    Tuple
)
from CNN.layers.layers import Layer, Trainable
from CNN.utils.data import get_batches
from tqdm import tqdm
import time


def evaluate(output, target):
    """
    Evaluate the accuracy of the model
    :param output:
    :param target:
    :return: accuracy in a scale of 0 to 1
    """
    return np.mean(np.argmax(target, axis=1) == np.argmax(output, axis=1))


class Model:
    """
    Model of neural network
    """

    def __init__(self, layers=None):
        """
        :param layers: model's layers
        """
        if layers is None:
            layers = []
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

    def set_loss(self, loss):
        self.loss = loss

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

        batch_amount = (len(X_train) - 1) // batch_size + 1
        iteration = 1
        for epoch in range(epochs):
            description = "Epoch #{} :".format(epoch + 1)
            pbar = tqdm(range(batch_amount))
            pbar.set_description(description)
            for x_batch, y_batch in get_batches(X_train, y_train, batch_size):
                x_pred = x_batch.copy()
                for layer in self.layers:
                    x_pred = layer.run(x_pred)

                dA = self.loss.calc_derivative(x_pred, y_batch)

                for layer in reversed(self.layers):
                    dA = layer.backprop(dA)

                for layer in self.layers:
                    if isinstance(layer, Trainable):
                        layer.update_params('adam', batch_size, **optimizer_params, t=iteration)

                pbar.update(1)

            print("Epoch {} : {}%".format(epoch, self.evaluate(X_test, y_test) * 100))
            iteration += batch_size

    def predict(self, batch):
        x_pred = batch.copy()
        for layer in self.layers:
            x_pred = layer.run(x_pred, is_training=False)
        return x_pred

    def evaluate(self, batch, labels):
        predictions = self.predict(batch)
        return evaluate(predictions, labels)
