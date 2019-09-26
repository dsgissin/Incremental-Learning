import numpy as np
from keras import backend as K
from keras.layers import Layer, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.initializers import Constant
from keras.callbacks import Callback
import argparse


class StopAtThreshold(Callback):

    def __init__(self,
                 monitor='loss',
                 threshold=0.001):
        super(StopAtThreshold, self).__init__()

        self.monitor = monitor
        self.threshold = threshold


    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if current < self.threshold:
            self.model.stop_training = True
            return


class RecordValues(Callback):

    def __init__(self, depth):
        self.depth = depth
        super(RecordValues, self).__init__()

    def on_epoch_end(self, epoch, logs=None):

        positive_w, negative_w = self.model.get_weights()
        W = positive_w**self.depth - negative_w**self.depth
        logs['weights'] = W

        return


class ToyModel(Layer):

    def __init__(self, d, depth, init_size, **kwargs):
        self.d = d
        self.depth = depth
        self.init = init_size**(1./depth)
        super(ToyModel, self).__init__(**kwargs)

    def build(self, input_shape):

        self.positive_kernel = self.add_weight(name="positive_w",
                                    shape=(d,),
                                    initializer=Constant(self.init),
                                    trainable=True)
        self.negative_kernel = self.add_weight(name="negative_w",
                                    shape=(d,),
                                    initializer=Constant(self.init),
                                    trainable=True)
        super(ToyModel, self).build(input_shape)

    def call(self, x, **kwargs):
        W = K.pow(self.positive_kernel, self.depth) - K.pow(self.negative_kernel, self.depth)
        return K.dot(x, K.reshape(W, (-1,1)))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


def OMP(x_train, y_train, verb=0):

    print("Running OMP...")

    indices = []
    weights = []
    weights.append(np.zeros(x_train.shape[1]))

    normalized_x = x_train / np.linalg.norm(x_train, axis=0, keepdims=True)

    r = y_train
    for i in range(normalized_x.shape[1]):
        if verb>0:
            print("Iteration {i}...".format(i=i+1))

        # find the next index:
        idx = np.argmax(np.abs(normalized_x.T @ r.squeeze()))
        if 0.5*np.linalg.norm(r)**2 < 0.0001:  # if the loss is small enough
            break
        if idx in indices:  # if we've reached zero loss and an index is picked twice
            break
        indices.append(idx)

        # find the optimal weights:
        w = np.zeros(normalized_x.shape[1],)
        w_idx = np.linalg.lstsq(normalized_x[:,np.array(indices)], y_train.squeeze(), rcond=None)[0]
        w[np.array(indices)] = w_idx

        # update the residuals:
        y_hat = normalized_x @ w
        r = y_train.squeeze() - y_hat.squeeze()

        # get the weights of the un-normalized model:
        w = np.zeros(normalized_x.shape[1],)
        w_idx = np.linalg.lstsq(x_train[:,np.array(indices)], y_train.squeeze(), rcond=None)[0]
        w[np.array(indices)] = w_idx
        weights.append(w)

    return indices, weights[-1]


def run_toy_model(x_train, y_train, depth=5, init=0.0003, epochs=100000, lr=0.003, verb=2, threshold=0.01):

    print("Running Toy Model...")

    # create the model:
    i = Input((x_train.shape[1],))
    out = ToyModel(d=x_train.shape[1], depth=depth, init_size=init)(i)
    model = Model(inputs=i, outputs=out)
    model.compile(SGD(lr), loss='mse')

    # train the model:
    hist = model.fit(x=x_train, y=y_train,
                       batch_size=x_train.shape[0],
                       epochs=epochs,
                       shuffle=False,
                       callbacks=[StopAtThreshold(threshold=threshold),
                                  RecordValues(depth=depth)],
                       verbose=verb)

    weights = hist.history['weights'][-1]
    if len(hist.history['weights']) == epochs:
        print("Possibly Not Converged!")

    # extracting the indices (index chosen when first larger then 0.1 in absolute value):
    w = np.array(hist.history['weights'])
    activations = np.zeros(w.shape)
    activations[w > 0.1] = 1
    idx = np.argmax(activations, axis=0)
    non_idx_count = np.sum(idx == 0)
    idx_count = x_train.shape[1] - non_idx_count
    idx[idx == 0] = epochs+1
    indices = list(np.argsort(idx)[:idx_count])

    return indices, weights


def check_sets(omp_set, gd_set):
    return float(len(omp_set.intersection(gd_set))) / float(min(len(omp_set), len(gd_set)))


def parse_input():
    p = argparse.ArgumentParser()
    p.add_argument('sparsity', type=int, default=5)
    p.add_argument('lr', type=float, default=0.001)
    p.add_argument('init_size', type=float, default=0.0001)
    p.add_argument('epochs', type=int, default=100000)
    p.add_argument('d', type=int, default=1000)
    p.add_argument('depth', type=int, default=5)
    p.add_argument('data_size', type=int, default=80)
    p.add_argument('verbosity', type=int, default=0)
    p.add_argument('convergence_threshold', type=float, default=0.01)
    args = p.parse_args()
    return args


if __name__ == "__main__":

    args = parse_input()

    epochs = args.epochs
    d = args.d
    data_size = args.data_size
    sparsity = args.sparsity
    lr = args.lr
    init = args.init_size
    verb = args.verbosity
    threshold = args.convergence_threshold

    opt_w = []
    omp_w = []
    omp_idx = []
    bp_w = []
    gd_w = []
    gd_idx = []

    # create the optimal vector:
    Wstar = np.zeros((d,1))
    indices = np.random.choice(d, sparsity, replace=False)
    values = np.random.exponential(10, (sparsity,1)) + 1
    Wstar[indices] = values

    # create the dataset:
    x_train = np.random.randn(data_size, d)
    x_train /= np.linalg.norm(x_train, axis=0, keepdims=True)
    y_train = x_train @ Wstar

    # run OMP on the data:
    omp_indices, omp_weights = OMP(x_train, y_train, verb=verb)

    # run the toy model on the data:
    gd_indices, gd_weights = run_toy_model(x_train, y_train, depth=args.depth, init=init, lr=lr, epochs=epochs, verb=verb, threshold=threshold)

    # calculate the concurrence of OMP and GD:
    print("The concurrence between the models is: " + str(check_sets(set(omp_indices), set(gd_indices))))