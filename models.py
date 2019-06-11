import keras
import math
from keras import callbacks
from keras.layers import Dense, Dropout, Conv1D, Flatten, Activation, MaxPooling1D,GlobalAveragePooling1D, AvgPool1D, BatchNormalization, GaussianNoise
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.initializers import TruncatedNormal, Constant
from keras.constraints import MaxNorm, UnitNorm
from keras.regularizers import l2
from keras import optimizers
from helpers import EnforceNeg, count_lines, generate_name_from_params
import numpy as np
import os
import tensorflow as tf
from parameters_complete import TEST_DIR
import multiprocessing
import h5py
from tensorflow.python import debug as tf_debug

PREFIX = os.environ['PREFIX']




def create_dense_model(train_indices, test_indices, params):
    """
    Params: 
    batch_size
    dropout_rate
    epochs
    """
    model = Sequential()

    model.add(Flatten(input_shape=(10020, 3)))

    model.add(Dense(units=1,
                    activation='sigmoid',
                    kernel_regularizer=keras.regularizers.l1(l=params['reg_rate']),
                    bias_constraint=EnforceNeg()))  # Negative bias are crucial for LRP


    sgd = optimizers.SGD(lr=params['learning_rate'], decay=params['decay'], momentum=params['momentum'], nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.summary()
    # Callbacks
    MODEL_NAME = generate_name_from_params(params)

    training_generator = DataGenerator(train_indices, train_indices,
                                            categorical=False,
                                            feature_matrix_path=params['feature_matrix_path'],
                                            y_path=params['y_path'])

    testing_generator = DataGenerator(test_indices, test_indices,
                                            categorical=False,
                                            feature_matrix_path=params['feature_matrix_path'],
                                            y_path=params['y_path'])

    tensorboardCb = callbacks.TensorBoard(log_dir=os.path.join(TEST_DIR,'tb',PREFIX,MODEL_NAME), histogram_freq=0)

    
    # Model fitting
    history = model.fit_generator(generator=training_generator,
                                  validation_data=testing_generator,
                                  use_multiprocessing=False,
                                  #workers=6,
                                  epochs=params['epochs'],
                                  verbose=params['verbose'],
                                  callbacks=[
                                    tensorboardCb,
                                  ])
    model.summary()

    return history, model




def create_conv_model(train_indices, test_indices, params):
    

    #keras.backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "node10:6064"))
    print(params)
    
    
    # VGG like
    model = Sequential()
    #model.add(GaussianNoise(params['noise'], input_shape=(10020, 3)))
    
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(10020, 3)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
  
    
    model.add(Dense(1, activation='sigmoid'))

  


    sgd = optimizers.SGD(lr=params['learning_rate'], decay=params['decay'], momentum=params['momentum'], nesterov=False)

    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    print(model.summary())


    MODEL_NAME = generate_name_from_params(params)
    training_generator = DataGenerator(train_indices, train_indices,
                                        categorical=False,
                                        feature_matrix_path=params['feature_matrix_path'],
                                        y_path=params['y_path'])

    testing_generator = DataGenerator(test_indices, test_indices,
                                            categorical=False,
                                            feature_matrix_path=params['feature_matrix_path'],
                                            y_path=params['y_path'])

    tensorboardCb = callbacks.TensorBoard(log_dir=os.path.join(TEST_DIR,'tb',PREFIX,MODEL_NAME), histogram_freq=0)


    # Model fitting
    history = model.fit_generator(generator=training_generator,
                                  validation_data=testing_generator,
                                  use_multiprocessing=False,
                                  #workers=6,
                                  epochs=params['epochs'],
                                  verbose=params['verbose'],
                                  callbacks=[
                                    tensorboardCb
                                  ])

    return history, model



class DataGenerator(keras.utils.Sequence):
    """ This generator streams batches of data to our models. All they need is a set of indexes to extract from the 
        SAVED FEATURE MATRIX.
    """
    

    def __init__(self, x_indexes, labels_indexes,feature_matrix_path, y_path, categorical=True, batch_size=32, dim=(10020,3), shuffle=True):
        
        self.x_path = feature_matrix_path
        self.y_path = y_path
        self.dim = dim
        self.batch_size = batch_size
        self.x_indexes = x_indexes
        self.shuffle = shuffle
        self.categorical = categorical
        self.on_epoch_end() # Shuffle data!

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
        # Generate data
        indices = list(np.sort(indices))
        with h5py.File(self.x_path,'r') as f:
            X = f['X'][indices]
        with h5py.File(self.y_path,'r') as l :
            y = l['X'][indices]
        
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
            y = keras.utils.to_categorical(y, num_classes=2)
        return X, y


