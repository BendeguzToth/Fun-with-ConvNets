import numpy as np


class VectorCrossEntropy:
    @staticmethod
    def error(result, label):
        return - label * np.log(result + 1e-7)

    @staticmethod
    def gradient(result, label):
        return - label / (result + 1e-7)


class SGD:
    def __init__(self, layers, learning_rate:float):
        for layer in layers:
            layer.assign_optimizer(self)
        self.learning_rate = learning_rate

    def __call__(self, layer, grad):
        for var_idx in range(len(layer.vars)):
            layer.vars[var_idx] -= self.learning_rate * grad[var_idx]


class Adam:
    """
    This class implements the Adam optimizer.
    """
    def __init__(self, layers, learning_rate=lambda n: 0.001, beta_1=0.9, beta_2=0.999):
        self.layers = layers
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m = {layer: [0 for _ in layer.vars] for layer in self.layers}
        self.v = {layer: [0 for _ in layer.vars] for layer in self.layers}
        self.n = {layer: 1 for layer in self.layers}

        for layer in self.layers:
            layer.assign_optimizer(self)

    def __call__(self, layer, grad):
        for var_idx in range(len(layer.vars)):
            self.m[layer][var_idx] = self.beta_1 * self.m[layer][var_idx] + (1 - self.beta_1) * grad[var_idx]
            self.v[layer][var_idx] = self.beta_2 * self.v[layer][var_idx] + (1 - self.beta_2) * grad[var_idx] ** 2
            # Bias correction.
            m_hat = self.m[layer][var_idx] / (1 - self.beta_1 ** self.n[layer])
            v_hat = self.v[layer][var_idx] / (1 - self.beta_2 ** self.n[layer])
            # Update.
            layer.vars[var_idx] -= self.learning_rate(self.n[layer]) * m_hat / (np.sqrt(v_hat) + 1e-7)
        self.n[layer] += 1


class RMSprop:
    def __init__(self, layers, learning_rate=lambda n: 0.001, decay_rate=0.95, epsilon=1e-7):
        self.layers = layers
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.n = {layer: 1 for layer in self.layers}
        self.cache = {layer: [0. for _ in layer.vars] for layer in self.layers}

        for layer in self.layers:
            layer.assign_optimizer(self)

    def __call__(self, layer, grad):
        for var_idx in range(len(layer.vars)):
            self.cache[layer][var_idx] = self.decay_rate * self.cache[layer][var_idx] + (1 - self.decay_rate) * grad[var_idx] ** 2
            layer.vars[var_idx] -= self.learning_rate(self.n[layer]) * grad[var_idx] / (np.sqrt(self.cache[layer][var_idx] + self.epsilon))
        self.n[layer] += 1


class IncrementalAverage:
    """
    This class enables us to store the running average
    of some datapoints on an incremental way, without
    the need of saving all the values.
    """
    def __init__(self):
        self.counter = 1
        self.sum  = 0

    def __str__(self):
        return str(self.sum)

    def __repr__(self):
        return "[counter: {0}, sum: {1}]".format(self.counter - 1, self.sum)

    def add(self, x):
        """
        Add a new data point.
        :param x: New data point to be added to the average.
        """
        self.sum += (1 / self.counter) * (x - self.sum)
        self.counter += 1

    def get(self):
        """
        :return: The average.
        """
        return self.sum

    def reset(self):
        """
        Clears the cache.
        """
        self.counter = 1
        self.sum = 0