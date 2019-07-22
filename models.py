from lrp import ExLinear, ExReLU, ExSequential, ExDropout, ExConv1d
import torch.nn.functional as F
import torch.nn
import h5py
from parameters_complete import TEST_DIR, SAVED_MODELS_DIR, Cs, n_total_snps, random_state
import os
import numpy as np
import math

import tensorflow
import keras

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Conv1D, Flatten, Activation, AveragePooling1D, MaxPooling1D, GlobalAveragePooling1D, AvgPool1D, BatchNormalization, GaussianNoise
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.initializers import TruncatedNormal, Constant
from keras.constraints import MaxNorm, UnitNorm
from keras.regularizers import l2, l1, l1_l2
from keras import optimizers
from helpers import EnforceNeg, generate_name_from_params
keras.constraints.EnforceNeg = EnforceNeg  # Absolutely crucial

PREFIX = os.environ['PREFIX']


best_params_montavon = {
            'epochs': 400,
            'l1_reg': 1e-6,
            'l2_reg': 1e-1,
            'dropout_rate': 0.3,
            'optimizer':  'adam',
            'patience': 7,
            'kernel_window_size': 30,

}
def create_montavon_conv_model(params):
        model = Sequential()

        model.add(Conv1D(activation='relu',
                        input_shape=(10020, 3),
                        filters=10,
                        kernel_size=params['kernel_window_size'],
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], l2=params['l2_reg']),
                        ))

        model.add(Dropout(params['dropout_rate']))

        model.add(Flatten())


        model.add(Dense(activation='sigmoid',
                        units=1,
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], l2=params['l2_reg'])
                        ))

        model.compile(loss='binary_crossentropy',
                        optimizer=params['optimizer'],
                        metrics=['accuracy'])


        return model


def create_lenet_model(params):
        model = Sequential()

        model.add(Conv1D(activation='relu',
                        input_shape=(30060, 1),
                        filters=6,
                        strides=5,
                        kernel_size=params['kernel_window_size'],
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        kernel_initializer=keras.initializers.glorot_normal(),
                        bias_constraint=EnforceNeg()
                        ))
            
        model.add(AveragePooling1D(2))

        model.add(Conv1D(activation='relu',
                        filters=16,
                        kernel_size=5,
                        strides=1,
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        kernel_initializer=keras.initializers.glorot_normal(),
                        bias_constraint=EnforceNeg()
                        ))
            
        model.add(AveragePooling1D(2))

        
        model.add(Flatten())

        model.add(Dense(activation='relu',
                        units=10,
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        bias_constraint=EnforceNeg()))

        model.add(Dense(activation='sigmoid',
                        units=1,
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        bias_constraint=EnforceNeg()))


        model.compile(loss='binary_crossentropy',
                        optimizer=params['optimizer'],
                        metrics=['accuracy'])


        return model


def create_clairvoyante_model(params):
        model = Sequential()

        model.add(Conv1D(activation='relu',
                        input_shape=(30060, 1),
                        filters=16,
                        strides=1,
                        kernel_size=params['kernel_window_size'],
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        kernel_initializer=keras.initializers.glorot_normal(),
                        bias_constraint=EnforceNeg()
                        ))
            
        model.add(AveragePooling1D(5))

        model.add(Conv1D(activation='relu',
                        filters=32,
                        kernel_size=5,
                        strides=1,
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        kernel_initializer=keras.initializers.glorot_normal(),
                        bias_constraint=EnforceNeg()
                        ))
            
        model.add(AveragePooling1D(2))

        
        model.add(Flatten())

        model.add(Dense(activation='relu',
                        units=10,
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        bias_constraint=EnforceNeg()))

        model.add(Dense(activation='sigmoid',
                        units=1,
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        bias_constraint=EnforceNeg()))


        model.compile(loss='binary_crossentropy',
                        optimizer=params['optimizer'],
                        metrics=['accuracy'])


        return model


def create_explanable_conv_model(params):
        model = Sequential()

        model.add(Conv1D(activation='relu',
                        input_shape=(30060, 1),
                        filters=params['filter_nb'],
                        strides=5,
                        kernel_size=params['kernel_window_size'],
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        kernel_initializer=keras.initializers.glorot_normal(),
                        bias_constraint=EnforceNeg()
                        ))
            
        model.add(Conv1D(activation='relu',
                        filters=5,
                        kernel_size=3,
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        kernel_initializer=keras.initializers.glorot_normal(),
                        bias_constraint=EnforceNeg()
                        ))
        model.add(AveragePooling1D(2))

        
        model.add(Flatten())


        model.add(Dense(activation='sigmoid',
                        units=1,
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        bias_constraint=EnforceNeg()))


        model.compile(loss='binary_crossentropy',
                        optimizer=params['optimizer'],
                        metrics=['accuracy'])


        return model


best_params_montaez = {
                'epochs': 500,
                'l1_reg': 10e-5,
                'l2_reg': 10e-3,
                'dropout_rate': 0.3,
                'batch_size': 32,
                'optimizer': 'adam',
                'patience':20
}
def create_montaez_dense_model(params):

    model=Sequential()
    model.add(Flatten(input_shape=(10020, 3)))

    model.add(Dense(activation='relu',
                    units=10,
                    kernel_regularizer=l1_l2(
                        l1=params['l1_reg'], l2=params['l2_reg'])
                    ))

    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(activation='relu',
                    units=10,
                    kernel_regularizer=l1_l2(
                        l1=params['l1_reg'], l2=params['l2_reg'])
                    ))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(activation='sigmoid',
                    units=1,
                    kernel_regularizer=l1_l2(
                        l1=params['l1_reg'], l2=params['l2_reg'])
                    ))

    model.compile(loss='binary_crossentropy',
                    optimizer=params['optimizer'],
                    metrics=['accuracy'])


    return model





def train_dummy_dense_model(features, labels, indices, params):

        model=Sequential()
        model.add(Flatten(input_shape=(10020, 3)))

        model.add(Dense(activation='relu',
                        units=2,
                        bias_constraint=EnforceNeg()  # Negative bias are crucial for LRP


                        ))
        model.add(Activation('softmax'))


        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])


        cb_chkpt=ModelCheckpoint(params['saved_model_path'], monitor='val_loss', verbose=0,
                                   save_best_only=True, save_weights_only=False, mode='min', period=1)

        # Model fitting
        history=model.fit(x=features[indices.train], y=labels[indices.train],
                            validation_data=(
                                features[indices.test], labels[indices.test]),
                            epochs=params['epochs'],
                            verbose=0,
                            callbacks=[
                                cb_chkpt
                            ])
        score=max(history.history['val_acc'])
        print("DNN max val_acc: {}".format(score))
        return load_model(params['saved_model_path'])

def create_dense_model(train_indices, test_indices, params):
    """
    Params:
    batch_size
    dropout_rate
    epochs
    """
    model=Sequential()

    model.add(Flatten(input_shape=(10020, 3)))

    model.add(Dense(units=1,
                    activation='sigmoid',
                    kernel_regularizer=keras.regularizers.l1(
                        l=params['reg_rate']),
                    bias_constraint=EnforceNeg()))  # Negative bias are crucial for LRP


    sgd=optimizers.SGD(lr=params['learning_rate'], decay=params['decay'],
                       momentum=params['momentum'], nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.summary()

    MODEL_NAME=generate_name_from_params(params)

    training_generator=DataGenerator(train_indices, train_indices,
                                            categorical=False,
                                            feature_matrix_path=params['feature_matrix_path'],
                                            y_path=params['y_path'])

    testing_generator=DataGenerator(test_indices, test_indices,
                                            categorical=False,
                                            feature_matrix_path=params['feature_matrix_path'],
                                            y_path=params['y_path'])

    tensorboardCb=callbacks.TensorBoard(log_dir=os.path.join(
        TEST_DIR, 'tb', PREFIX, MODEL_NAME), histogram_freq=0)


    # Model fitting
    history=model.fit_generator(generator=training_generator,
                                  validation_data=testing_generator,
                                  use_multiprocessing=False,
                                  # workers=6,
                                  epochs=params['epochs'],
                                  verbose=params['verbose'],
                                  callbacks=[
                                    tensorboardCb,
                                  ])
    model.summary()

    return history, model



class DataGenerator(keras.utils.Sequence):
    """ This generator streams batches of data to our models. All they need is a set of indexes to extract from the
        SAVED FEATURE MATRIX.
    """


    def __init__(self, x_indexes, labels_indexes, feature_matrix_path, y_path, categorical=True, batch_size=32, dim=(10020, 3), shuffle=True):

        self.x_path=feature_matrix_path
        self.y_path=y_path
        self.dim=dim
        self.batch_size=batch_size
        self.x_indexes=x_indexes
        self.shuffle=shuffle
        self.categorical=categorical
        self.on_epoch_end()  # Shuffle data!

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x_indexes) / float(self.batch_size)))

    def __getitem__(self, index):
        # Generate indexes for each batch
        indexes=self.x_indexes[index*self.batch_size:(index+1)*self.batch_size]

        return self.__data_generation(indexes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            random_state.shuffle(self.x_indexes)


    def __data_generation(self, indices):
        # X : (n_samples, *dim, n_channels)
        'Generates data from SAVED FEATURE MATRIX containing batch_size samples'
        # Initialization
        assert len(indices) == self.batch_size
        # Generate data
        indices=list(np.sort(indices))
        with h5py.File(self.x_path, 'r') as f:
            X=f['X'][indices]
        with h5py.File(self.y_path, 'r') as l:
            y=l['X'][indices]

        """
        counter = 0
        with open(self.y_path, 'r') as fp:
            for line_nb, line in enumerate(fp):
                    if counter >= self.batch_size:
                        # Found every index
                        break
                    if line_nb == indices[counter]:
                        y[counter] = 0 if (int(line)<0) else 1
                        counter+=1

        """
        assert X.shape == (self.batch_size, *self.dim)
        assert y.shape[0] == self.batch_size
#        assert counter == self.batch_size
        if self.categorical:
            y=keras.utils.to_categorical(y, num_classes=2)
        return X, y

class MontaezEarlyStopping(EarlyStopping):

    """ Performs early stopping by monitoring reductions of val_loss by percentage
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):

        self.min_delta = - self.best * 0.01
        super().on_epoch_end(epoch, logs)
