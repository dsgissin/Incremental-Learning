import numpy as np
from keras import backend as K
from keras.layers import Layer, Input, Activation
from keras.models import Model
from keras.optimizers import SGD
from keras.initializers import RandomNormal, Identity, Orthogonal, Zeros, Constant, Ones
from keras.callbacks import Callback
from sklearn.svm import LinearSVC
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

        w = get_canonical_model(self.model, self.depth)
        w = np.abs(w)
        top_5_idx = np.argsort(w)[-5:]
        top_5_values = [w[i] for i in top_5_idx]
        logs['unnormalized_ev1'] = top_5_values[0]
        logs['unnormalized_ev2'] = top_5_values[1]
        logs['unnormalized_ev3'] = top_5_values[2]
        logs['unnormalized_ev4'] = top_5_values[3]
        logs['unnormalized_ev5'] = top_5_values[4]

        w /= np.linalg.norm(w)
        top_5_idx = np.argsort(w)[-5:]
        top_5_values = [w[i] for i in top_5_idx]
        logs['ev1'] = top_5_values[0]
        logs['ev2'] = top_5_values[1]
        logs['ev3'] = top_5_values[2]
        logs['ev4'] = top_5_values[3]
        logs['ev5'] = top_5_values[4]

        return


class Separator(Layer):

    def __init__(self, d, init, depth, **kwargs):
        """
        The initializer for the linear deparator class (neural network with diagonal weight matrices)
        :param d: the input dimension
        :param init: the scale of initialization (sigme_0)
        :param depth: the model's depth
        """
        self.d = d
        self.init = init
        self.depth = depth
        super(Separator, self).__init__(**kwargs)

    def build(self, input_shape):

        self.w_plus = self.add_weight(name='w_plus',
                                    shape=(self.d, ),
                                     initializer=Constant(init ** (1./self.depth)),
                                     trainable=True)

        self.w_minus = self.add_weight(name='w_minus',
                                    shape=(self.d, ),
                                     initializer=Constant(init ** (1./self.depth)),
                                     trainable=True)

        super(Separator, self).build(input_shape)

    def call(self, x, **kwargs):

        w_plus = K.pow(self.w_plus, self.depth)
        w_minus = K.pow(self.w_minus, self.depth)

        return K.dot(x, K.reshape(w_plus - w_minus, (-1,1)))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


def get_canonical_model(model, depth):
    w_plus = model.get_weights()[0]
    w_minus = model.get_weights()[1]
    return w_plus**depth - w_minus**depth


def exp_loss(y_true, y_pred):
    return K.exp((1-2*y_true) * y_pred)


def parse_input():
    p = argparse.ArgumentParser()
    p.add_argument('lr', type=float, default=0.01)
    p.add_argument('init_size', type=float, default=0.01)
    p.add_argument('epochs', type=int, default=100000)
    p.add_argument('d', type=int, default=100)
    p.add_argument('data_size', type=int, default=200)
    p.add_argument('convergence_threshold', type=float, default=0.01)
    p.add_argument('loss', type=float, default=True, help="True for exponential loss, False for Logistic loss.")
    p.add_argument('verbosity', type=int, default=2)
    args = p.parse_args()
    return args


if __name__ == "__main__":

    args = parse_input()
    lr = args.lr
    epochs = args.epochs
    init = args.init_size
    d = args.d
    data_size = args.data_size
    threshold = args.convergence_threshold
    exponential_loss = args.loss
    verbosity = args.verbosity

    # generate the normalized sparse labeling function:
    Wstar = np.zeros((d,))
    Wstar[0] = 8
    Wstar[1] = 4
    Wstar[2] = 2
    Wstar[3] = 1
    Wstar /= np.linalg.norm(Wstar)


    # create the dataset:
    x_train = np.random.randn(data_size, d)
    logits = x_train @ Wstar
    y_train = np.sign(logits)
    y_train[y_train<0] = 0


    # create the models:
    i1 = Input((d,))
    out1 = Separator(d=d, init=init, depth=1)(i1)
    if exponential_loss:
        model1 = Model(inputs=i1, outputs=out1)
        model1.compile(SGD(lr), loss=exp_loss)
    else:
        out1 = Activation('sigmoid')(out1)
        model1 = Model(inputs=i1, outputs=out1)
        model1.compile(SGD(lr), loss='binary_crossentropy', metrics=['acc'])

    i2 = Input((d,))
    out2 = Separator(d=d, init=init, depth=2)(i2)
    if exponential_loss:
        model2 = Model(inputs=i2, outputs=out2)
        model2.compile(SGD(lr,), loss=exp_loss)
    else:
        out2 = Activation('sigmoid')(out2)
        model2 = Model(inputs=i2, outputs=out2)
        model2.compile(SGD(lr), loss='binary_crossentropy', metrics=['acc'])

    i3 = Input((d,))
    out3 = Separator(d=d, init=init, depth=3)(i3)
    if exponential_loss:
        model3 = Model(inputs=i3, outputs=out3)
        model3.compile(SGD(lr), loss=exp_loss)
    else:
        out3 = Activation('sigmoid')(out3)
        model3 = Model(inputs=i3, outputs=out3)
        model3.compile(SGD(lr), loss='binary_crossentropy', metrics=['acc'])

    i4 = Input((d,))
    out4 = Separator(d=d, init=init, depth=4)(i4)
    if exponential_loss:
        model4 = Model(inputs=i4, outputs=out4)
        model4.compile(SGD(lr), loss=exp_loss)
    else:
        out4 = Activation('sigmoid')(out4)
        model4 = Model(inputs=i4, outputs=out4)
        model4.compile(SGD(lr), loss='binary_crossentropy', metrics=['acc'])

    i5 = Input((d,))
    out5 = Separator(d=d, init=init, depth=5)(i5)
    if exponential_loss:
        model5 = Model(inputs=i5, outputs=out5)
        model5.compile(SGD(lr), loss=exp_loss)
    else:
        out5 = Activation('sigmoid')(out5)
        model5 = Model(inputs=i5, outputs=out5)
        model5.compile(SGD(lr), loss='binary_crossentropy', metrics=['acc'])

    model6 = LinearSVC()

    # train the models:
    print("Training Depth 1 Model...")
    hist1 = model1.fit(x=x_train, y=y_train,
              batch_size=data_size,
              epochs = epochs,
              shuffle=False,
              callbacks=[StopAtThreshold(threshold=threshold), RecordValues(depth=1)],
              verbose = verbosity)
    print("Training Depth 2 Model...")
    hist2 = model2.fit(x=x_train, y=y_train,
              batch_size=data_size,
              epochs = epochs,
              shuffle=False,
              callbacks=[StopAtThreshold(threshold=threshold), RecordValues(depth=2)],
              verbose = verbosity)
    print("Training Depth 3 Model...")
    hist3 = model3.fit(x=x_train, y=y_train,
              batch_size=data_size,
              epochs = epochs,
              shuffle=False,
              callbacks=[StopAtThreshold(threshold=threshold), RecordValues(depth=3)],
              verbose = verbosity)
    print("Training Depth 4 Model...")
    hist4 = model4.fit(x=x_train, y=y_train,
              batch_size=data_size,
              epochs = epochs,
              shuffle=False,
              callbacks=[StopAtThreshold(threshold=threshold), RecordValues(depth=4)],
              verbose = verbosity)
    print("Training Depth 5 Model...")
    hist5 = model5.fit(x=x_train, y=y_train,
              batch_size=data_size,
              epochs = epochs,
              shuffle=False,
              callbacks=[StopAtThreshold(threshold=threshold), RecordValues(depth=5)],
              verbose = verbosity)

    print("Training SVM Model...")
    model6.fit(x_train, y_train)

    # extract and normalize the final weight vectors:
    w1 = get_canonical_model(model1, 1)
    w2 = get_canonical_model(model2, 2)
    w3 = get_canonical_model(model3, 3)
    w4 = get_canonical_model(model4, 4)
    w5 = get_canonical_model(model5, 5)
    w6 = model6.coef_
    w1 /= np.linalg.norm(w1)
    w2 /= np.linalg.norm(w2)
    w3 /= np.linalg.norm(w3)
    w4 /= np.linalg.norm(w4)
    w5 /= np.linalg.norm(w5)
    w6 /= np.linalg.norm(w6)

    # print the correlation to the labeling function:
    print("W1 Correlation:  " + str(Wstar @ w1))
    print("W2 Correlation:  " + str(Wstar @ w2))
    print("W3 Correlation:  " + str(Wstar @ w3))
    print("W4 Correlation:  " + str(Wstar @ w4))
    print("W5 Correlation:  " + str(Wstar @ w5))
    print("SVM Correlation: " + str(Wstar @ w6.squeeze()))