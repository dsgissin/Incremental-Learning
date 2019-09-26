import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Layer, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.initializers import RandomNormal, Identity, Orthogonal, Zeros, Constant
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

    def __init__(self, bias=True):
        self.bias = bias
        super(RecordValues, self).__init__()

    def on_epoch_end(self, epoch, logs=None):

        W = self.model.get_weights()[0]
        W = W.T @ W

        vals = get_sorted_eig(W)
        logs['ev1'] = vals[0]
        logs['ev2'] = vals[1]
        logs['ev3'] = vals[2]
        logs['ev4'] = vals[3]
        logs['ev5'] = vals[4]

        if self.bias:
            b = self.model.get_weights()[1]
            logs['b'] = b

        return


class Quadratic(Layer):

    def __init__(self, d, init_size, bias=True, bias_init=0, **kwargs):
        self.d = d
        self.init = init_size**0.5
        self.bias = bias
        self.bias_init = bias_init
        super(Quadratic, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(name='W',
                                    shape=(self.d, self.d),
                                     initializer=Orthogonal(gain=self.init),
                                     trainable=True)

        if self.bias:
            self.b = self.add_weight(name='b',
                                            shape=(1,),
                                            initializer=Constant(self.bias_init),
                                            trainable=True)

        super(Quadratic, self).build(input_shape)

    def call(self, x, **kwargs):
        pre_activation = K.dot(x, K.transpose(self.W))
        activation = K.square(pre_activation)

        if self.bias:
            return K.sum(activation, axis=1, keepdims=True) + self.b
        else:
            return K.sum(activation, axis=1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


def get_sorted_eig(W):
    vals = np.linalg.svd(W, True, False)
    return vals


def parse_input():
    p = argparse.ArgumentParser()
    p.add_argument('lr', type=float, default=0.001)
    p.add_argument('init_size', type=float, default=0.0001)
    p.add_argument('epochs', type=int, default=100000)
    p.add_argument('d', type=int, default=20)
    p.add_argument('data_size', type=int, default=100)
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
    init = args.init_size
    threshold = args.convergence_threshold
    verbosity = args.verbosity

    # initialize the low rank optimal quadratic network:
    Wstar = np.zeros((d,d))
    Wstar[0,0] = 1. ** 0.5
    Wstar[1,1] = (1./2.) ** 0.5
    Wstar[2,2] = (1./3.) ** 0.5
    Wstar[3,3] = (1./4.) ** 0.5
    b_star = 0

    # create the dataset:
    x_train = np.random.randn(data_size, d)
    y_train = np.sum(np.square(x_train @ Wstar.T), axis=1) + b_star
    b_init = np.mean(y_train)


    # create the models:
    i1 = Input((d,))
    out1 = Quadratic(d=d, init_size=init, bias=False)(i1)
    model1 = Model(inputs=i1, outputs=out1)
    model1.compile(SGD(lr), loss='mse')

    i2 = Input((d,))
    out2 = Quadratic(d=d, init_size=init, bias=True, bias_init=b_star+b_init)(i2)
    model2 = Model(inputs=i2, outputs=out2)
    model2.compile(SGD(lr), loss='mse')


    # train the models:
    hist1 = model1.fit(x=x_train, y=y_train,
              batch_size=data_size,
              epochs = epochs,
              shuffle=False,
              callbacks=[StopAtThreshold(threshold=threshold), RecordValues(bias=False)],
              verbose = verbosity)
    hist2 = model2.fit(x=x_train, y=y_train,
              batch_size=data_size,
              epochs = epochs,
              shuffle=False,
              callbacks=[StopAtThreshold(threshold=threshold), RecordValues(bias=True)],
              verbose = verbosity)


    # plot incremental learning:
    plt.figure()
    plt.subplot(121)
    plt.plot(np.arange(len(hist1.history['ev1'])), hist1.history['ev1'])
    plt.plot(np.arange(len(hist1.history['ev2'])), hist1.history['ev2'])
    plt.plot(np.arange(len(hist1.history['ev3'])), hist1.history['ev3'])
    plt.plot(np.arange(len(hist1.history['ev4'])), hist1.history['ev4'])
    plt.plot(np.arange(len(hist1.history['ev5'])), hist1.history['ev5'])
    plt.ylabel("Eigenvalue")
    plt.xlabel("Iteration")
    plt.title("Without Bias")
    plt.subplot(122)
    plt.plot(np.arange(len(hist2.history['ev1'])), hist2.history['ev1'])
    plt.plot(np.arange(len(hist2.history['ev2'])), hist2.history['ev2'])
    plt.plot(np.arange(len(hist2.history['ev3'])), hist2.history['ev3'])
    plt.plot(np.arange(len(hist2.history['ev4'])), hist2.history['ev4'])
    plt.plot(np.arange(len(hist2.history['ev5'])), hist2.history['ev5'])
    # plt.plot(np.arange(len(hist2.history['b'])), hist2.history['b'], ':')
    plt.ylabel("Eigenvalue")
    plt.xlabel("Iteration")
    plt.title("With Bias")
    plt.show()