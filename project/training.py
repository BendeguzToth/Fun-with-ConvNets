"""
WARNING
It takes a really long time to train, not worth it.
"""

import numpy as np
from layers import Convolution, MaxPooling, ReLU, Reshape, Dense, Softmax,  Dropout, BatchNorm
from network import Network
from utils import IncrementalAverage, VectorCrossEntropy, Adam
from cifar10.cifar_loader import load_cifar10

# Path to the unzipped CIFAR-10 folder
DATA_PATH = ""
# Will print progress message after number of batches is completed.
LOG_FREQ = 100
STARTING_EPOCH = 1
EPOCHS = 100
BATCH_SIZE = 50

training_data, training_labels, test_data, test_labels, _ = load_cifar10(DATA_PATH, normalize=True)


def make_batch(data, label, size):
    for start, end in zip(range(0, len(data), size), range(size, len(data)+1, size)):
        yield data[start:end, ...], label[start:end, ...]


network = Network(
    Convolution(input_shape=(32, 32), input_depth=3, n_filters=32, filter_dim=(3, 3), stride=(1, 1), padding=((1, 1), (1, 1))),
    ReLU(),
    BatchNorm(),
    Convolution(input_shape=(32, 32), input_depth=32, n_filters=32, filter_dim=(3, 3), stride=(1, 1), padding=((1, 1), (1, 1))),
    ReLU(),
    BatchNorm(),
    MaxPooling(input_shape=(32, 32), input_depth=32, filter_dim=(2, 2), stride=(2, 2)),
    Dropout(rate=0.2),

    Convolution(input_shape=(16, 16), input_depth=32, n_filters=64, filter_dim=(3, 3), stride=(1, 1), padding=((1, 1), (1, 1))),
    ReLU(),
    BatchNorm(),
    Convolution(input_shape=(16, 16), input_depth=64, n_filters=64, filter_dim=(3, 3), stride=(1, 1), padding=((1, 1), (1, 1))),
    ReLU(),
    BatchNorm(),
    MaxPooling(input_shape=(16, 16), input_depth=64, filter_dim=(2, 2), stride=(2, 2)),
    Dropout(rate=0.3),

    Convolution(input_shape=(8, 8), input_depth=64, n_filters=128, filter_dim=(3, 3), stride=(1, 1), padding=((1, 1), (1, 1))),
    ReLU(),
    BatchNorm(),
    Convolution(input_shape=(8, 8), input_depth=128, n_filters=128, filter_dim=(3, 3), stride=(1, 1), padding=((1, 1), (1, 1))),
    ReLU(),
    BatchNorm(),
    MaxPooling(input_shape=(8, 8), input_depth=128, filter_dim=(2, 2), stride=(2, 2)),
    Dropout(rate=0.4),

    Reshape(input_shape=(128, 4, 4), output_shape=(2048, 1)),
    Dense(size=10, input_len=2048),
    Softmax()
)

optimizer = Adam(network.trainables, learning_rate=lambda n: 0.0001, beta_1=0.9, beta_2=0.999)

avg = IncrementalAverage()
for epoch in range(STARTING_EPOCH, STARTING_EPOCH + EPOCHS):
    batch = 1
    for x, y in make_batch(training_data, training_labels, BATCH_SIZE):
        out = network(x)
        avg.add(np.sum(VectorCrossEntropy.error(out, y)))
        network.backward(VectorCrossEntropy.gradient(out, y), update=True)
        if batch % LOG_FREQ == 0:
            print(f"epoch {epoch}/{EPOCHS} | batch {batch} - loss: {avg.get()}")
        batch += 1
    # Testing
    testacc = IncrementalAverage()
    # Split the test data into 10 batches in order to fit in RAM.
    for testbatch in range(0, 10000, 1000):
        testacc.add(network.accuracy(test_data[testbatch:testbatch + 1000], test_labels[testbatch:testbatch + 1000]))
    accuracy = testacc.get()
    print(f"Test accuracy : {accuracy}")
    # network.save_weights("path-to-savefile")
    print(f"Epoch {epoch} done =================")
    avg.reset()
