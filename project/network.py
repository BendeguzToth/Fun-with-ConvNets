from functools import reduce
import numpy as np
from layers import Trainable
import json


class Network:
    def __init__(self, *layers):
        self.layers = layers

    @property
    def trainables(self):
        return list(filter(lambda layer: issubclass(type(layer), Trainable), self.layers))

    def num_params(self):
        return reduce(lambda acc, param: acc + np.product(param.shape), self.trainables)

    def save_weights(self, file):
        with open(file, 'w') as file:
            json.dump(reduce(lambda lst, trainable: lst + [trainable.save()], self.trainables, []), file)

    def load(self, file):
        with open(file, 'r') as file:
            lst = json.load(file)
        list(map(lambda trainable, data: trainable.load(data), self.trainables, lst))

    def __call__(self, x, training=True):
        return reduce(lambda val, layer: layer(val, training), self.layers, x)

    def backward(self, gradient, update):
        return reduce(lambda grad, layer: layer.backward(grad, update), self.layers[::-1], gradient)

    def accuracy(self, x, labels):
        """
        Single class classification, one-hot column vector output.
        """
        res = self(x, training=False)
        clean = np.zeros(res.shape)
        clean[np.arange(x.shape[0]), np.argmax(res, axis=1).flatten()] = 1
        return np.sum(clean * labels) / x.shape[0]