import numpy as np
import json


def indices(Fh, Fw, Sh, Sw, steps_h, steps_w):
    """
    Returns indices for the im2col operation.
    :param Fh: Filter height.
    :param Fw: Filter width.
    :param Sh: Vertical stride.
    :param Sw: Horizontal stride.
    :param steps_h: Height of output.
    :param steps_w: Width of output.
    :return: tuple of indices.
    """
    # We make the first column
    index_x = np.arange(Fh)[..., None]
    index_x = np.repeat(index_x, Fw, axis=0)

    # The x-coordinates will be the same when doing width steps.
    # The number of different x-coordinate columns is the number
    # of height steps. First we make only the different columns,
    # and we will repeat them afterwards.
    index_x = np.repeat(index_x, steps_h, axis=1)

    # Now every column needs to be added to the stride along
    # height. So if the stride is 2, we need to add [0, 2, 4, ...]
    index_x += np.arange(steps_h) * Sh

    # Now we only need to copy it a few times. For every width
    # step the x coordinates don't change, so we need to repeat
    # it steps_along_w times.
    index_x = np.repeat(index_x, steps_w, axis=1)

    # Now we make the y indices. First we begin with a single row.
    # For every row, we have a range of length filter_size_w,
    # which pattern is the same for all the rows. There are
    # filter_size_h rows.
    index_y = np.arange(Fw)[..., None]
    index_y = np.tile(index_y, (Fh, 1))

    # There are width step different values, each height step times.
    index_y = np.repeat(index_y, steps_w, axis=1)

    # Again, each filter along width will be different by the stride.
    index_y += np.arange(steps_w) * Sw

    # Now we copy it for every height movement.
    index_y = np.tile(index_y, (1, steps_h))
    return index_x, index_y


class Trainable:
    def __init__(self):
        self.optimizer = None

    def assign_optimizer(self, optimizer):
        self.optimizer = optimizer

    @property
    def vars(self):
        return []

    def save(self):
        return list(map(lambda var: var.tolist() if type(var) is np.ndarray else var, self.vars))

    def load(self, lst):
        ...


class Convolution(Trainable):
    def __init__(self, input_shape, input_depth, n_filters, filter_dim, stride, padding=((0, 0), (0, 0))):
        Trainable.__init__(self)
        self.filter_dim = filter_dim
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.n_filters = n_filters
        # We store the filters as column vectors.
        self.filter = np.random.randn(n_filters, input_depth, 1, np.product(filter_dim))
        # Number of steps along (axis=0, axis=1) == (H, W)
        self.steps = (input_shape[0] + padding[0][0] + padding[0][1] - filter_dim[0]) // stride[0] + 1,\
                     (input_shape[1] + padding[1][0] + padding[1][1] - filter_dim[1]) // stride[1] + 1
        self.padding = padding
        self.unpad_indices = (slice(None), slice(None)) + tuple(map(lambda x: slice(x[0], None if x[1] == 0 else -x[1]), padding))
        self.indices = indices(*filter_dim, *stride, *self.steps)
        self.bias = np.random.randn(n_filters, )
        self.cache = None

    @property
    def vars(self):
        return [self.filter, self.bias]

    def load(self, lst):
        self.filter = np.array(lst[0])
        self.bias = np.array(lst[1])

    def __call__(self, x, training):
        x = np.pad(x, pad_width=((0, 0), (0, 0), *self.padding), constant_values=0)
        self.cache = x[..., self.indices[0], self.indices[1]]
        x = np.einsum('ndof,bdfi->bnoi', self.filter, self.cache)
        x += self.bias[..., None, None]
        return np.reshape(x, newshape=(-1, self.n_filters, *self.steps))

    def backward(self, gradient, update=False):
        """
        :param gradient: (BxDxHxW)
        :param update: Optimizer will be called if set to True.
        :return: dL/dInput
        """
        d_b = np.sum(np.sum(gradient, axis=(2, 3)), axis=0)

        gradient = np.reshape(gradient, newshape=(gradient.shape[0], gradient.shape[1], 1, np.product(gradient.shape[-2:])))
        d_w = np.einsum('bdfi,bnzi->ndzf', self.cache, gradient)

        d_input = np.einsum('ndof,bnoi->bdfi', self.filter, gradient)
        mem = np.zeros((gradient.shape[0], self.input_depth, self.input_shape[0] + np.sum(self.padding[0]), self.input_shape[1] + np.sum(self.padding[1])))
        np.add.at(mem, (slice(None), slice(None), *self.indices), d_input)
        if update:
            self.optimizer(self, [d_w, d_b])

        return mem[self.unpad_indices]


class MaxPooling:
    def __init__(self, input_shape, input_depth, filter_dim, stride, padding=((0, 0), (0, 0))):
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.padding = padding
        self.unpad_indices = (slice(None), slice(None)) + tuple(
            map(lambda x: slice(x[0], None if x[1] == 0 else -x[1]), padding))
        self.steps = (input_shape[0] + padding[0][0] + padding[0][1] - filter_dim[0]) // stride[0] + 1, \
                     (input_shape[1] + padding[1][0] + padding[1][1] - filter_dim[1]) // stride[1] + 1
        self.indices = indices(*filter_dim, *stride, *self.steps)
        self.cache = None
        self.out = None

    def __call__(self, x, training):
        batch_dim, depth = x.shape[0], x.shape[1]
        x = np.pad(x, pad_width=((0, 0), (0, 0), *self.padding), constant_values=0)
        self.cache = x[..., self.indices[0], self.indices[1]]
        self.out = np.max(self.cache, axis=2, keepdims=True)
        return np.reshape(self.out, newshape=(batch_dim, depth, *self.steps))

    def backward(self, gradient, update=False):
        batch_size = gradient.shape[0]
        gradient = np.reshape(gradient,
                              newshape=(gradient.shape[0], gradient.shape[1], 1, np.product(gradient.shape[-2:])))
        unwrapped_grad = gradient * (self.cache == self.out)
        mem = np.zeros((batch_size, self.input_depth, self.input_shape[0] + np.sum(self.padding[0]), self.input_shape[1] + np.sum(self.padding[1])))
        np.add.at(mem, (slice(None), slice(None), *self.indices), unwrapped_grad)
        return mem[self.unpad_indices]


class BatchNorm(Trainable):
    def __init__(self, momentum=0.99):
        Trainable.__init__(self)
        self.gamma = np.array([1.])
        self.beta = np.array([0.])
        self.inference_mean = np.array([0.])
        self.inference_variance = np.array([1.])
        self.momentum = momentum
        self.epsilon = 1e-7
        self.inner = None
        self.x_min_mean = None
        self.denom = None
        self.norm = None

    def __call__(self, x, training):
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            self.x_min_mean = x - np.broadcast_to(mean, x.shape)
            self.inner = np.broadcast_to(var, x.shape) + self.epsilon
            self.denom = 1 / np.sqrt(self.inner)
            self.norm = self.x_min_mean * self.denom
            self.inference_mean = self.momentum * self.inference_mean + (1 - self.momentum) * mean
            self.inference_variance = self.momentum * self.inference_variance + (1 - self.momentum) * var
            return self.gamma * self.norm + self.beta
        else:
            return self.gamma * ((x - self.inference_mean) / np.sqrt(self.inference_variance + self.epsilon)) + self.beta

    @property
    def vars(self):
        return [self.gamma, self.beta]

    def save(self):
        return [self.gamma.tolist(), self.beta.tolist(), self.inference_mean.tolist(), self.inference_variance.tolist(), [self.momentum]]

    def load(self, lst):
        self.gamma = np.array(lst[0])
        self.beta = np.array(lst[1])
        self.inference_mean = np.array(lst[2])
        self.inference_variance = np.array(lst[3])
        self.momentum = lst[4][0]

    def backward(self, gradient, update):
        d_beta = np.sum(gradient)
        d_gamma = np.sum(gradient * self.norm)
        ungamma = gradient * self.gamma
        unpow1 = ungamma * self.x_min_mean * -1/2 * (self.inner ** (-3/2))
        unmean = np.full(shape=unpow1.shape, fill_value=1 / unpow1.shape[0]) * np.broadcast_to(np.sum(unpow1, axis=0), unpow1.shape)
        unpow2 = 2 * self.x_min_mean * unmean + ungamma * self.denom
        unmean2 = np.broadcast_to(np.sum(-unpow2, axis=0), unpow2.shape) * np.full(unpow2.shape, 1/unpow2.shape[0])
        if update:
            self.optimizer(self, [d_gamma, d_beta])
        return unmean2 + unpow2


class ReLU:
    def __init__(self):
        self.cache = None

    def __call__(self, x, training):
        self.cache = np.maximum(x, 0)
        return self.cache

    def backward(self, gradient, update=False):
        return gradient * (self.cache > 0)


class Reshape:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __call__(self, x, training):
        """
        Does not include batch dimensions.
        """
        return np.reshape(x, (-1,) + self.output_shape)

    def backward(self, gradient, update=False):
        return np.reshape(gradient, (-1,)+self.input_shape)


class Dense(Trainable):
    def __init__(self, size, input_len):
        Trainable.__init__(self)
        self.w = np.random.randn(size, input_len) * np.sqrt(1 / input_len)
        self.b = np.random.randn(size, 1)
        self.cache = None

    @property
    def vars(self):
        return [self.w, self.b]

    def load(self, lst):
        self.w = np.array(lst[0])
        self.b = np.array(lst[1])

    def __call__(self, x, training):
        self.cache = x
        return np.matmul(self.w, x) + self.b

    def backward(self, gradient, update=False):
        d_b = np.sum(gradient, axis=0)
        d_w = np.sum(np.matmul(gradient, np.swapaxes(self.cache, 1, 2)), axis=0)
        d_input = np.matmul(self.w.T, gradient)
        if update:
            self.optimizer(self, [d_w, d_b])

        return d_input


class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.rescale = 1 / (1 - self.rate)
        self.mask = None

    def __call__(self, x, training):
        if training:
            self.mask = np.random.random(x.shape) > self.rate
            return self.mask * x * self.rescale
        else:
            self.mask = np.ones_like(x)
            return x

    def backward(self, gradient, update=False):
        return gradient * self.mask * self.rescale


class Softmax:
    def __init__(self):
        """
        Dense. Operates on the last axis.
        """
        self.x = None
        self.exps = None
        self.S = None

    def __call__(self, x, training):
        shiftx = x - np.max(x, axis=1, keepdims=True)
        self.exps = np.exp(shiftx)
        self.S = np.sum(self.exps, axis=1, keepdims=True)
        self.x = self.exps / self.S
        return self.x

    def backward(self, gradient, update=False):
        local_grad = np.matmul(-self.x, np.transpose(self.x, axes=(0, 2, 1))) * np.repeat(
            np.expand_dims(1 - np.identity(self.x.shape[1]), axis=0), self.x.shape[0], axis=0) + (
                                 1 - self.x) * self.x * np.repeat(np.expand_dims(np.identity(self.x.shape[1]), axis=0),
                                                                  self.x.shape[0], axis=0)
        return np.matmul(local_grad, gradient)

