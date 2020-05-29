from CNN.model import Model
from CNN.layers.conv import Conv2D
from CNN.layers.pooling import MaxPooling2D
from CNN.layers.core import (
    Dropout,
    Flattening,
    Dense
)


def create_model():
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
    return model
