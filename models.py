from keras import callbacks
from keras.layers import Dense, Dropout, Conv1D, Flatten, Activation, AvgPool1D
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.initializers import TruncatedNormal, Constant
from keras.constraints import MaxNorm, UnitNorm
from helpers import EnforceNeg
import numpy as np
import os


def create_dense_model(x_train, y_train, x_test, y_test, params):
    """
    Params: 
    batch_size
    dropout_rate
    epochs
    """
    model = Sequential()
    model.add(Dense(units=2,
                    activation='relu',
                    kernel_initializer=TruncatedNormal(
                        mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=Constant(value=-0.0),
                    bias_constraint=EnforceNeg()
                    ))  # Negative bias are crucial
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Callbacks
    tensorboardCb = callbacks.TensorBoard(log_dir=dir_path+'/keras-logs', histogram_freq=0,
                                          write_graph=True, write_images=True)
    earlyStoppingCb = callbacks.EarlyStopping(
        patience=4, monitor='val_acc')

    # Model fitting
    history = model.fit(x_train, y_train,
                        epochs=params['epochs'],
                        batch_size=params['batch_size'],
                        verbose=0,
                        callbacks=[
                            tensorboardCb,
                            # earlyStoppingCb
                        ],
                        validation_data=[x_test, y_test])

    return history, model


def create_conv_model(x_train, y_train, x_test, y_test, params):

    model = Sequential()
    model.add(Conv1D(filters=1,
                     kernel_size=35,
                     strides=1,
                     padding='valid',
                     bias_initializer=Constant(value=-0.0),
                     bias_constraint=EnforceNeg()))  # n, d', 3
    model.add(Activation('relu'))

    model.add(AvgPool1D(pool_size=5, padding='valid'))  # n, d''/pool_size, 5

    model.add(Flatten())  # n, d''/pool_size * 5

    model.add(Dense(units=2,
                    bias_initializer=Constant(value=-0.0),
                    bias_constraint=EnforceNeg()))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Callbacks
    tensorboardCb = callbacks.TensorBoard(log_dir=dir_path+'/tb-logs', histogram_freq=0,
                                          write_graph=True, write_images=True)
    earlyStoppingCb = callbacks.EarlyStopping(
        patience=6, monitor='val_acc')

    # Model fitting
    history = model.fit(x_train, y_train,
                        epochs=params['epochs'],
                        batch_size=params['batch_size'],
                        verbose=0,
                        callbacks=[
                            tensorboardCb,
                            earlyStoppingCb
                        ],
                        validation_data=[x_test, y_test])

    print("NUMBER OF PARAMETERS: {}".format(model.count_params()))

    return history, model
