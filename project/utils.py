import numpy as np
import pickle


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


def load_cifar10(directory, normalize=True):
    """
    This function loads the CIFAR10 dataset
    from the batch files located in the directory.
    :param directory: Path to the directory where the
    batch files are located. It will search for the original
    file names, eg.: data_batch_1 - 5, test_batch.
    :param normalize: If True, the pixel values are divided by
    255. (All values are between 0 and 1.) If False, values are
    0 - 255.
    :return: Tuple of (training_data,
    training_labels, test_data, test_labels, label_names).
    The *_data entries will be numpy arrays with dimensions
    (number_of_images, depth (r, g, b), height, width), the *_labels
    will be one-hot column vectors with the correct classes:
    (number_of_labels, number_of_classes, 1).
    Label names is a list containing strings. The name that corresponds
    to each label is the string at that index.
    """
    training_data = []
    training_labels = []
    for i in range(1, 6):
        try:
            d = unpickle(directory + f"/data_batch_{i}")
        except FileNotFoundError:
            raise Exception(f"File 'data_batch_{i}' is not found in the specified directory '{directory}'.")
        training_data.append(d[b"data"])
        training_labels.append(d[b"labels"])
    training_data = np.vstack(training_data)
    training_data = np.reshape(training_data, newshape=(-1, 3, 32, 32))
    training_labels = np.concatenate(training_labels)
    training_labels = np.array(list(map(lambda hot: one_hot(10, hot), training_labels)))

    try:
        test = unpickle(directory + "/test_batch")
    except FileNotFoundError:
        raise Exception(f"File 'test_batch' is not found in the specified directory '{directory}'.")
    test_data = np.reshape(test[b"data"], newshape=(-1, 3, 32, 32))
    test_labels = np.array(list(map(lambda hot: one_hot(10, hot), test[b"labels"])))

    try:
        meta = unpickle(directory + "/batches.meta")
    except FileNotFoundError:
        raise Exception(f"File 'batches.meta' is not found in the specified directory '{directory}'.")
    label_names = meta[b"label_names"]
    label_names = list(map(lambda x: x.decode("utf-8"), label_names))

    if normalize:
        training_data  = training_data / 255
        test_data = test_data / 255

    return training_data, training_labels, test_data, test_labels, label_names


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def one_hot(size, hot):
    vec = np.zeros((size,))
    vec[hot] = 1
    return vec[..., None]
