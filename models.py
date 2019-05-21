import keras
import math
from keras import callbacks
from keras.layers import Dense, Dropout, Conv1D, Flatten, Activation, AvgPool1D, BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.initializers import TruncatedNormal, Constant
from keras.constraints import MaxNorm, UnitNorm
from helpers import EnforceNeg, count_lines
import numpy as np
import os
import multiprocessing


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


    training_generator = DataGenerator(x_train_indices, y_train_indices,
                                       feature_matrix_path=params['feature_matrix_path'],
                                       y_path=params['y_path'])
    testing_generator = DataGenerator(x_test_indices, y_test_indices,
                                      feature_matrix_path=params['feature_matrix_path'],
                                      y_path=params['y_path'])

    # Model fitting
    history = model.fit_generator(generator=training_generator,
                                  validation_data=testing_generator,
                                  use_multiprocessing=False,
                                  #workers=6,
                                  epochs=params['epochs'],
                                  verbose=params['verbose'],
                                  callbacks=[
                                    # tensorboardCb,
                                    # earlyStoppingCb
                                  ])

    return history, model


def create_conv_model(x_train_indices, y_train_indices, x_test_indices, y_test_indices, params):
    print(params)
    model = Sequential()
    model.add(Conv1D(filters=5, 
                     kernel_size=3,
                     strides=1,
                     padding='valid',
                     input_shape=(10020, 3),
                     bias_initializer=Constant(value=-0.0),
                     bias_constraint=EnforceNeg()))  # n, d', 3
    if params['use_normalization']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=params['dropout_rate']))
    model.add(AvgPool1D(pool_size=2, padding='valid'))  # n, d''/pool_size, 5

    model.add(Conv1D(filters=5,
                     kernel_size=3,
                     strides=1,
                     padding='valid',
                     bias_initializer=Constant(value=-0.0),
                     bias_constraint=EnforceNeg()))  # n, d', 3
    if params['use_normalization']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=params['dropout_rate']))
    model.add(AvgPool1D(pool_size=2, padding='valid'))  # n, d''/pool_size, 5

    model.add(Conv1D(filters=5,
                     kernel_size=3,
                     strides=1,
                     padding='valid',
                     bias_initializer=Constant(value=-0.0),
                     bias_constraint=EnforceNeg()))  # n, d', 3
    if params['use_normalization']:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=params['dropout_rate']))
    model.add(AvgPool1D(pool_size=2, padding='valid'))  # n, d''/pool_size, 5



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

    training_generator = DataGenerator(x_train_indices, y_train_indices,
                                       feature_matrix_path=params['feature_matrix_path'],
                                       y_path=params['y_path'])
    testing_generator = DataGenerator(x_test_indices, y_test_indices,
                                      feature_matrix_path=params['feature_matrix_path'],
                                      y_path=params['y_path'])

    # Model fitting
    history = model.fit_generator(generator=training_generator,
                                  validation_data=testing_generator,
                                  use_multiprocessing=False,
                                  #workers=6,
                                  epochs=params['epochs'],
                                  verbose=params['verbose'],
                                  callbacks=[
                                    # tensorboardCb,
                                    # earlyStoppingCb
                                  ])

    print("NUMBER OF PARAMETERS: {}".format(model.count_params()))

    return history, model



class DataGenerator(keras.utils.Sequence):
    """ This generator streams batches of data to our models. All they need is a set of indexes to extract from the 
        SAVED FEATURE MATRIX.
    """
    

    def __init__(self, x_indexes, labels_indexes,feature_matrix_path, y_path,  batch_size=32, dim=(10020,3), shuffle=True):
        
        self.x_path = feature_matrix_path
        self.y_path = y_path
        self.dim = dim
        self.batch_size = batch_size
        self.x_indexes = x_indexes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x_indexes) / float(self.batch_size)))

    def __getitem__(self, index):
        # Generate indexes for each batch
        indexes = self.x_indexes[index*self.batch_size:(index+1)*self.batch_size]

        return self.__data_generation(indexes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.x_indexes)
        

    def __data_generation(self, indices):
        'Generates data from SAVED FEATURE MATRIX containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        assert len(indices) == self.batch_size
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        indices = np.sort(indices) 
        X = np.load(self.x_path, mmap_mode='r') # Check this out !
        X = np.take(X, indices, axis=0)
        
        counter = 0
        with open(self.y_path, 'r') as fp:
            for line_nb, line in enumerate(fp):
                    if counter >= self.batch_size:
                        # Found every index
                        break
                    if line_nb == indices[counter]:
                        y[counter] = 0 if (int(line)<0) else 1
                        counter+=1 
  

        assert X.shape == (self.batch_size, *self.dim)
        assert y.shape[0] == self.batch_size
        assert counter == self.batch_size
        return X, keras.utils.to_categorical(y, num_classes=2)
