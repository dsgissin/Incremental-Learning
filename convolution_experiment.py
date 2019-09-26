import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Layer, Input, Activation
from keras.models import Model
from keras.optimizers import SGD
from keras.initializers import RandomNormal, Identity, Orthogonal, Zeros, Constant, Ones
from keras.callbacks import Callback
from keras.activations import sigmoid
from sklearn.svm import LinearSVC
import tensorflow as tf
import argparse

class StopAtThreshold(Callback):

    def __init__(self,
                 monitor='loss',
                 threshold=0.01):
        super(StopAtThreshold, self).__init__()

        self.monitor = monitor
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if current <= self.threshold:
            self.model.stop_training = True
            return


class RecordValues(Callback):

    def __init__(self, depth, metric):
        self.depth = depth
        self.metric = metric
        super(RecordValues, self).__init__()

    def on_epoch_end(self, epoch, logs=None):

        w = get_canonical_model(self.model)
        top_5_idx = np.argsort(np.abs(w))[-5:]
        top_5_values = [np.abs(w)[i] for i in top_5_idx]
        logs['unnormalized_ev1'] = top_5_values[0]
        logs['unnormalized_ev2'] = top_5_values[1]
        logs['unnormalized_ev3'] = top_5_values[2]
        logs['unnormalized_ev4'] = top_5_values[3]
        logs['unnormalized_ev5'] = top_5_values[4]

        w /= np.linalg.norm(w)
        top_5_idx = np.argsort(np.abs(w))[-5:]
        top_5_values = [np.abs(w)[i] for i in top_5_idx]
        logs['ev1'] = top_5_values[0]
        logs['ev2'] = top_5_values[1]
        logs['ev3'] = top_5_values[2]
        logs['ev4'] = top_5_values[3]
        logs['ev5'] = top_5_values[4]

        return


class DeepLinearConv(Layer):

    def __init__(self, d, depth, norm, **kwargs):
        self.d = d
        self.depth = depth
        self.norm = norm * (3./2.)
        self.const = self.norm**(1./depth)
        super(DeepLinearConv, self).__init__(**kwargs)

    def build(self, input_shape):

        weights = []
        for i in range(self.depth-1):
            weights.append(self.add_weight(name=str(i),
                                      shape=(self.d,1,1),
                                      initializer=RandomNormal(stddev=self.const*np.sqrt(self.d)**((self.depth-1)/self.depth)/self.d),
                                      trainable=True)
                                )

        weights.append(self.add_weight(name="last",
                                       shape=(self.d,1),
                                       initializer=RandomNormal(stddev=self.const * np.sqrt(self.d) ** ((self.depth - 1) / self.depth) / self.d),
                                       trainable=True)
                       )

        self.weight_list = weights
        super(DeepLinearConv, self).build(input_shape)


    def call(self, x, **kwargs):

        # run circular convolutions on x:
        out = x
        for i in range(len(self.weight_list)-1):
            out = K.reshape(out, (-1, self.d))
            out = K.reshape(K.concatenate([out, out], axis=1)[:, :-1], (-1,2*self.d - 1,1))
            out = K.conv1d(out, self.weight_list[i], padding='valid')

        # end with the fully connected layer:
        return K.dot(K.reshape(out,(-1, self.d)), self.weight_list[-1])


    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


def get_canonical_model(model):
    weights = model.get_weights()
    w_canonical = np.prod([np.fft.rfft(x.squeeze()) for x in weights], axis=0)
    return w_canonical


def exp_loss(y_true, y_pred):
    return K.exp((1-2*y_true) * y_pred)


def exponential_acc(y_true, y_pred):
    y_pred_sign = K.sign(y_pred)
    y_pred_sign += 1
    y_pred_sign /= 2
    return K.mean(K.cast(K.equal(y_pred_sign, y_true), 'float64'))


def parse_input():
    p = argparse.ArgumentParser()
    p.add_argument('lr', type=float, default=0.001)
    p.add_argument('init_size', type=float, default=0.01)
    p.add_argument('epochs', type=int, default=500000)
    p.add_argument('d', type=int, default=100)
    p.add_argument('data_size', type=int, default=200)
    p.add_argument('convergence_threshold', type=float, default=0.01)
    p.add_argument('loss', type=int, default=1, help="True for exponential loss, False for Logistic loss.")
    p.add_argument('iterations', type=int, default=9)
    p.add_argument('verbosity', type=int, default=2)
    args = p.parse_args()
    return args

if __name__ == '__main__':

    args = parse_input()
    lr = args.lr
    epochs = args.epochs
    init = args.init_size
    d = args.d
    data_size = args.data_size
    threshold = args.convergence_threshold
    exponential_loss = args.loss
    iterations = args.iterations
    verbosity = args.verbosity

    if exponential_loss:
        metric = exponential_acc
        metric_name = 'exponential_acc'
    else:
        metric = 'acc'
        metric_name = 'acc'


    # create the sparse optimal vector in frequency space:
    Wstar = np.zeros((d,))
    Wstar = np.fft.rfft(Wstar)
    Wstar[0] = 3
    Wstar[1] = 4
    Wstar[2] = 6
    Wstar[3] = 12
    Wstar = np.fft.irfft(Wstar)
    Wstar /= np.linalg.norm(Wstar)
    Wstar_fft = np.fft.rfft(Wstar)

    vectors = []
    for it in range(iterations):

        print("At Iteration " + str(it))

        # create the dataset:
        x_train = np.random.randn(data_size, d)
        logits = x_train @ Wstar
        y_train = np.sign(logits)
        y_train[y_train<0] = 0

        # create the models:
        i1 = Input((d,))
        out1 = DeepLinearConv(d=d, depth=1, norm=init)(i1)
        if exponential_loss:
            model1 = Model(inputs=i1, outputs=out1)
            model1.compile(SGD(lr), loss=exp_loss, metrics=[metric])
        else:
            out1 = Activation('sigmoid')(out1)
            model1 = Model(inputs=i1, outputs=out1)
            model1.compile(SGD(lr), loss='binary_crossentropy', metrics=[metric])

        i2 = Input((d,))
        out2 = DeepLinearConv(d=d, depth=2, norm=init)(i2)
        if exponential_loss:
            model2 = Model(inputs=i2, outputs=out2)
            model2.compile(SGD(lr), loss=exp_loss, metrics=[metric])
        else:
            out2 = Activation('sigmoid')(out2)
            model2 = Model(inputs=i2, outputs=out2)
            model2.compile(SGD(lr), loss='binary_crossentropy', metrics=[metric])

        i3 = Input((d,))
        out3 = DeepLinearConv(d=d, depth=3, norm=init)(i3)
        if exponential_loss:
            model3 = Model(inputs=i3, outputs=out3)
            model3.compile(SGD(lr), loss=exp_loss, metrics=[metric])
        else:
            out3 = Activation('sigmoid')(out3)
            model3 = Model(inputs=i3, outputs=out3)
            model3.compile(SGD(lr), loss='binary_crossentropy', metrics=[metric])

        model4 = LinearSVC()


        # train the models:
        hist1 = model1.fit(x=x_train, y=y_train,
                  batch_size=data_size,
                  epochs = epochs,
                  shuffle=False,
                  callbacks=[StopAtThreshold(threshold=threshold), RecordValues(depth=1, metric=metric_name)],
                  verbose = verbosity)
        hist2 = model2.fit(x=x_train, y=y_train,
                  batch_size=data_size,
                  epochs = epochs,
                  shuffle=False,
                  callbacks=[StopAtThreshold(threshold=threshold), RecordValues(depth=2, metric=metric_name)],
                  verbose = verbosity)
        hist3 = model3.fit(x=x_train, y=y_train,
                  batch_size=data_size,
                  epochs = epochs,
                  shuffle=False,
                  callbacks=[StopAtThreshold(threshold=threshold), RecordValues(depth=3, metric=metric_name)],
                  verbose = verbosity)

        model4.fit(x_train, y_train)


        # extract the weights of the models:
        w1 = get_canonical_model(model1)
        w2 = get_canonical_model(model2)
        w3 = get_canonical_model(model3)
        w4 = model4.coef_
        w4 = np.real(np.fft.rfft(w4))

        # normalize the vectors:
        w1 /= np.linalg.norm(w1)
        w2 /= np.linalg.norm(w2)
        w3 /= np.linalg.norm(w3)
        w4 /= np.linalg.norm(w4)

        vectors.append([w1, w2, w3, w4])

    print(vectors)
