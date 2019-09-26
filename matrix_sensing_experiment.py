import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Layer, Input
from keras.models import Model
from keras.optimizers import SGD, Adam, Adadelta
from keras.initializers import RandomNormal, Identity, Orthogonal
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

    def __init__(self):
        super(RecordValues, self).__init__()

    def on_epoch_end(self, epoch, logs=None):

        # get the current induced linear model:
        W = get_canonical_model(self.model)

        # record the leading eigenvalues:
        vals = get_sorted_eig(W)
        logs['ev1'] = vals[0]
        logs['ev2'] = vals[1]
        logs['ev3'] = vals[2]
        logs['ev4'] = vals[3]
        logs['ev5'] = vals[4]

        return


class DeepLinear(Layer):

    def __init__(self, d, depth, init_size, init_type, **kwargs):
        self.d = d
        self.depth = depth
        self.init_size = init_size
        self.init_type = init_type
        # self.const = norm**(1./depth)
        super(DeepLinear, self).__init__(**kwargs)

    def build(self, input_shape):
        weights = []
        for i in range(self.depth):
            if self.init_type == 'Gaussian':
                weights.append(self.add_weight(name=str(i),
                                          shape=(d, d),
                                          # initializer=RandomNormal(stddev=self.const*np.sqrt(self.d)**((self.depth-1)/self.depth)/self.d),
                                          initializer=RandomNormal(stddev=self.init_size**(1./self.depth) * (1./np.sqrt(self.d))),
                                          trainable=True)
                                    )
            elif self.init_type == 'Identity':
                weights.append(self.add_weight(name=str(i),
                                          shape=(d, d),
                                          initializer=Identity(gain=self.init_size ** (1./self.depth)),
                                          trainable=True)
                                    )
        self.weight_list = weights
        super(DeepLinear, self).build(input_shape)

    def call(self, x, **kwargs):
        W = self.weight_list[0]
        for i in range(1, len(self.weight_list)):
            W = K.dot(W, self.weight_list[i])
        return K.dot(x, K.reshape(W, (-1,1)))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


def get_canonical_model(model):
    weights = model.get_weights()
    W = np.eye(d)
    for w in weights:
        W = W @ w
    return W


def get_sorted_eig(W):
    vals = np.linalg.svd(W, True, False)
    return vals


def parse_input():
    p = argparse.ArgumentParser()
    p.add_argument('lr', type=float, default=0.001)
    p.add_argument('init_size', type=float, default=0.0001)
    p.add_argument('init_type', type=str, default="Gaussian", choices={"Gaussian","Identity"})
    p.add_argument('epochs', type=int, default=100000)
    p.add_argument('d', type=int, default=50)
    p.add_argument('data_size', type=int, default=1000)
    p.add_argument('convergence_threshold', type=float, default=0.01)
    p.add_argument('verbosity', type=int, default=2)
    args = p.parse_args()
    return args


if __name__ == "__main__":

    args = parse_input()
    lr = args.lr
    epochs = args.epochs
    d = args.d
    data_size = args.data_size
    init_size = args.init_size
    init_type = args.init_type
    threshold = args.convergence_threshold
    verbosity = args.verbosity

    # create the optimal matrix:
    Wstar = np.zeros((d,d))
    Wstar[0,0] = 12
    Wstar[1,1] = 6
    Wstar[2,2] = 4
    Wstar[3,3] = 3

    # create the dataset:
    x_train = np.random.randn(data_size, d**2)
    y_train = x_train.reshape((data_size, -1)) @ Wstar.reshape((-1,1))

    # create the models:
    i1 = Input((d**2,))
    out1 = DeepLinear(d=d, depth=1, init_size=init_size, init_type=init_type)(i1)
    model1 = Model(inputs=i1, outputs=out1)
    model1.compile(SGD(lr), loss='mse')

    i2 = Input((d**2,))
    out2 = DeepLinear(d=d, depth=2, init_size=init_size, init_type=init_type)(i2)
    model2 = Model(inputs=i2, outputs=out2)
    model2.compile(SGD(lr), loss='mse')

    i3 = Input((d**2,))
    out3 = DeepLinear(d=d, depth=3, init_size=init_size, init_type=init_type)(i3)
    model3 = Model(inputs=i3, outputs=out3)
    model3.compile(SGD(lr), loss='mse')

    i4 = Input((d**2,))
    out4 = DeepLinear(d=d, depth=4, init_size=init_size, init_type=init_type)(i4)
    model4 = Model(inputs=i4, outputs=out4)
    model4.compile(SGD(lr), loss='mse')


    # print model norm: TODO
    W = get_canonical_model(model1)
    print("Model max eigenvalue:")
    print(np.max(np.linalg.eig(W)[0]))
    W = get_canonical_model(model2)
    print("Model max eigenvalue:")
    print(np.max(np.linalg.eig(W)[0]))
    W = get_canonical_model(model3)
    print("Model max eigenvalue:")
    print(np.max(np.linalg.eig(W)[0]))
    W = get_canonical_model(model4)
    print("Model max eigenvalue:")
    print(np.max(np.linalg.eig(W)[0]))

    # train the models:
    hist1 = model1.fit(x=x_train, y=y_train,
              batch_size=data_size,
              epochs = epochs,
              shuffle=False,
              callbacks=[StopAtThreshold(threshold=threshold), RecordValues()],
              verbose = verbosity)
    hist2 = model2.fit(x=x_train, y=y_train,
              batch_size=data_size,
              epochs = epochs,
              shuffle=False,
              callbacks=[StopAtThreshold(threshold=threshold), RecordValues()],
              verbose = verbosity)
    hist3 = model3.fit(x=x_train, y=y_train,
              batch_size=data_size,
              epochs = epochs,
              shuffle=False,
              callbacks=[StopAtThreshold(threshold=threshold), RecordValues()],
              verbose = verbosity)
    hist4 = model4.fit(x=x_train, y=y_train,
              batch_size=data_size,
              epochs = epochs,
              shuffle=False,
              callbacks=[StopAtThreshold(threshold=threshold), RecordValues()],
              verbose = verbosity)


    # print matrix reconstruction error:
    print("Final Reconstruction Errors:")
    W = get_canonical_model(model1)
    print("Depth 1: " + str(np.linalg.norm(W-Wstar)**2 / np.linalg.norm(Wstar)**2))
    W = get_canonical_model(model2)
    print("Depth 2: " + str(np.linalg.norm(W-Wstar)**2 / np.linalg.norm(Wstar)**2))
    W = get_canonical_model(model3)
    print("Depth 3: " + str(np.linalg.norm(W-Wstar)**2 / np.linalg.norm(Wstar)**2))
    W = get_canonical_model(model4)
    print("Depth 4: " + str(np.linalg.norm(W-Wstar)**2 / np.linalg.norm(Wstar)**2))

    # plot incremental learning of singular values:
    plt.figure()
    plt.subplot(221)
    plt.plot(np.arange(len(hist1.history['ev1'])), hist1.history['ev1'])
    plt.plot(np.arange(len(hist1.history['ev2'])), hist1.history['ev2'])
    plt.plot(np.arange(len(hist1.history['ev3'])), hist1.history['ev3'])
    plt.plot(np.arange(len(hist1.history['ev4'])), hist1.history['ev4'])
    plt.plot(np.arange(len(hist1.history['ev5'])), hist1.history['ev5'])
    plt.ylabel("Eigenvalue")
    plt.title("Depth 1")
    plt.subplot(222)
    plt.plot(np.arange(len(hist2.history['ev1'])), hist2.history['ev1'])
    plt.plot(np.arange(len(hist2.history['ev2'])), hist2.history['ev2'])
    plt.plot(np.arange(len(hist2.history['ev3'])), hist2.history['ev3'])
    plt.plot(np.arange(len(hist2.history['ev4'])), hist2.history['ev4'])
    plt.plot(np.arange(len(hist2.history['ev5'])), hist2.history['ev5'])
    plt.title("Depth 2")
    plt.subplot(223)
    plt.plot(np.arange(len(hist3.history['ev1'])), hist3.history['ev1'])
    plt.plot(np.arange(len(hist3.history['ev2'])), hist3.history['ev2'])
    plt.plot(np.arange(len(hist3.history['ev3'])), hist3.history['ev3'])
    plt.plot(np.arange(len(hist3.history['ev4'])), hist3.history['ev4'])
    plt.plot(np.arange(len(hist3.history['ev5'])), hist3.history['ev5'])
    plt.ylabel("Eigenvalue")
    plt.title("Depth 3")
    plt.subplot(224)
    plt.plot(np.arange(len(hist4.history['ev1'])), hist4.history['ev1'])
    plt.plot(np.arange(len(hist4.history['ev2'])), hist4.history['ev2'])
    plt.plot(np.arange(len(hist4.history['ev3'])), hist4.history['ev3'])
    plt.plot(np.arange(len(hist4.history['ev4'])), hist4.history['ev4'])
    plt.plot(np.arange(len(hist4.history['ev5'])), hist4.history['ev5'])
    plt.ylabel("Eigenvalue")
    plt.title("Depth 4")
    plt.show()
