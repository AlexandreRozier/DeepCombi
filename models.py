import os

import keras
import keras.callbacks
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Conv1D, Flatten, Activation, AveragePooling1D, MaxPooling1D
from keras.models import Sequential, load_model
from keras.regularizers import l1_l2
from talos.utils.gpu_utils import multi_gpu, parallel_gpu_jobs

from helpers import get_available_gpus

import pdb



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




best_convdense_params = {
    'epochs': 0,
    'kernel_nb':32,
    'strides': 3,
    'kernel_size': 3, #wtf
    'l1_reg':1e-4,
    'l2_reg':1e-6,
    'lr':0.001
}
def create_convdense_model(params):
    model = Sequential()

    model.add(Conv1D(activation='relu',
                    input_shape=(10020, 3),
                    filters=params['kernel_nb'],
                    strides=params['strides'],
                    kernel_size=params['kernel_size'],
                    kernel_regularizer=l1_l2(
                        l1=params['l1_reg'], 
                        l2=params['l2_reg']
                    ),
                    kernel_initializer=keras.initializers.glorot_normal(),

                    ))
        

    model.add(Flatten())

    
    model.add(Dense(activation='sigmoid',
                    units=1,
                    kernel_regularizer=l1_l2(
                        l1=params['l1_reg'], 
                        l2=params['l2_reg']
                    ),
                    ))


    model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.Adam(lr=params['lr']),
                    metrics=['accuracy'])


    return model

def create_lenet_model(params):
    model = Sequential()

    model.add(Conv1D(activation='relu',
                    input_shape=(30060, 1),
                    filters=params['kernel_nb'],
                    strides=2,
                    kernel_size=params['kernel_size'],
                    kernel_regularizer=l1_l2(
                        l1=params['l1_reg'], 
                        l2=params['l2_reg']
                    ),
                    kernel_initializer=keras.initializers.glorot_normal(),

                    ))
        
    model.add(MaxPooling1D(params['pooling_ratio']))

    model.add(Conv1D(activation='relu',
                    filters=params['kernel_nb']*2,
                    kernel_size=params['kernel_size'],
                    strides=2,
                    kernel_regularizer=l1_l2(
                        l1=params['l1_reg'], 
                        l2=params['l2_reg']
                    ),
                    kernel_initializer=keras.initializers.glorot_normal(),

                    ))
        
    model.add(MaxPooling1D(params['pooling_ratio']))


    model.add(Conv1D(activation='relu',
                    filters=params['kernel_nb']*4 ,
                    kernel_size=params['kernel_size'],
                    strides=2,
                    kernel_regularizer=l1_l2(
                        l1=params['l1_reg'], 
                        l2=params['l2_reg']
                    ),
                    kernel_initializer=keras.initializers.glorot_normal(),

                    ))
        
    model.add(MaxPooling1D(params['pooling_ratio']))


    model.add(Conv1D(activation='relu',
                    filters=params['kernel_nb']*8,
                    kernel_size=params['kernel_size'],
                    strides=2,
                    kernel_regularizer=l1_l2(
                        l1=params['l1_reg'], 
                        l2=params['l2_reg']
                    ),
                    kernel_initializer=keras.initializers.glorot_normal(),

                    ))
        
    model.add(MaxPooling1D(params['pooling_ratio']))


    model.add(Flatten())

    
    model.add(Dense(activation='softmax',
                    units=2,
                    kernel_regularizer=l1_l2(
                        l1=params['l1_reg'], 
                        l2=params['l2_reg']
                    ),
                    ))


    model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(lr=params['lr']),
                    metrics=['accuracy'])


    return model


def create_clairvoyante_model(params):
        model = Sequential()

        model.add(Conv1D(activation='relu',
                        input_shape=(30060, 1),
                        filters=32,
                        strides=1,
                        kernel_size=params['kernel_window_size'],
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        kernel_initializer=keras.initializers.glorot_normal(),

                        ))
            
        model.add(AveragePooling1D(4))

        model.add(Conv1D(activation='relu',
                        filters=48,
                        kernel_size=4,
                        strides=1,
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        kernel_initializer=keras.initializers.glorot_normal(),

                        ))
            
        model.add(AveragePooling1D(3))

        
        model.add(Flatten())

        model.add(Dense(activation='relu',
                        units=336,
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        ))

        model.add(Dense(activation='sigmoid',
                        units=84,
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        ))
        model.add(Dense(activation='sigmoid',
                        units=1,
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        ))


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

                        ))
            
        model.add(Conv1D(activation='relu',
                        filters=5,
                        kernel_size=3,
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        kernel_initializer=keras.initializers.glorot_normal(),

                        ))
        model.add(AveragePooling1D(2))

        
        model.add(Flatten())


        model.add(Dense(activation='sigmoid',
                        units=1,
                        kernel_regularizer=l1_l2(
                            l1=params['l1_reg'], 
                            l2=params['l2_reg']
                        ),
                        ))


        model.compile(loss='binary_crossentropy',
                        optimizer=params['optimizer'],
                        metrics=['accuracy'])


        return model


best_params_montaez = {
    'epochs': 500,
#    'batch_size': 32,   
    'l1_reg': 1e-4,
    'l2_reg': 1e-6,
    'lr' : 0.01,
    'dropout_rate':0.3,
    'factor':0.7125,
    'patience':50,
    'hidden_neurons':64
}


def create_montaez_dense_model(params):
    model=Sequential()
    model.add(Flatten(input_shape=(int(params['n_snps']), 3)))

    model.add(Dense(activation='relu', units=int(params['hidden_neurons']), kernel_regularizer=l1_l2(l1=params['l1_reg'], l2=params['l2_reg'])))

    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(activation='relu', units=int(params['hidden_neurons']), kernel_regularizer=l1_l2(l1=params['l1_reg'], l2=params['l2_reg'])))
	
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(activation='softmax', units=2, kernel_regularizer=l1_l2(l1=params['l1_reg'], l2=params['l2_reg'])))
	
    nb_gpus = get_available_gpus()
    if nb_gpus ==1:
        parallel_gpu_jobs(0.5)
    if nb_gpus >= 2:
        model = multi_gpu(model, gpus=get_available_gpus())
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=params['lr']),  weighted_metrics=['categorical_accuracy'], metrics=['categorical_accuracy'])
    #tf.keras.metrics.AUC(), tf.keras.metrics.AUC(curve='PR')
    return model



def train_dummy_dense_model(features, labels, indices, params):

        model=Sequential()
        model.add(Flatten(input_shape=(10020, 3)))

        model.add(Dense(activation='relu',
                        units=2,

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
