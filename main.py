from CNN.test import unit_test
from CNN.model import Model
from CNN.layers.conv import Conv2D
from CNN.layers.pooling import MaxPooling2D
from CNN.utils.loss import CategoricalCrossEntropy
from CNN.layers.core import (
    Dropout,
    Flattening,
    Dense
)
from data_preprocess import get_data

TEST = False


def test():
    unit_test()


def main():
    model = Model()
    model.add(Conv2D(32, (3, 3), 'relu', 'he_normal', 'zeros', 1))
    model.add(MaxPooling2D((2, 2), 2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), 'relu', 'glorot_uniform', 'zeros', 1))
    model.add(MaxPooling2D((2, 2), 2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), 'relu', 'glorot_uniform', 'zeros', 1))
    model.add(Dropout(0.4))
    model.add(Flattening())
    model.add(Dense(128, 'relu', 'glorot_uniform', 'zeros'))
    model.add(Dropout(0.3))
    model.add(Dense(2, 'softmax', 'glorot_uniform', 'zeros'))
    model.set_loss(CategoricalCrossEntropy)

    data = get_data()
    model.train(data, 32, 100, 'adam', lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-7)


if __name__ == '__main__':
    if TEST:
        test()
    else:
        main()
