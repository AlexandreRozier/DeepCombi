
import math
import os

import tensorflow 
from helpers import chi_square, string_to_featmat, EnforceNeg, generate_name_from_params
keras.constraints.EnforceNeg = EnforceNeg # Absolutely crucial

import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split, KFold

from parameters_complete import (Cs, classy, filter_window_size,
                                 p_pnorm_filter, pnorm_feature_scaling,
                                 svm_rep, TEST_DIR, TALOS_OUTPUT_DIR, DATA_DIR, PARAMETERS_DIR)
from models import create_conv_model, create_dense_model

class TestQsub(object):
    

    np.random.seed(int(os.environ['SGE_TASK_ID']))

    def test_generate_conf(self):

        """ Generates a conf file containing a hyperparams set for each QSUB Job
        """
        CONF_PATH = os.path.join(PARAMETERS_DIR,os.environ['SGE_TASK_ID'])

        try:
            os.remove(CONF_PATH)
        except Exception as e:
            print(e)

        params_range = {
            'epochs': [500],
            #'dropout_rate': [0],
            'batch_size': np.linspace(32, 500, 10),
            'feature_matrix_path': [os.path.join(DATA_DIR,'3d_feature_matrix.npy')],
            'y_path':[os.path.join(DATA_DIR,'syn_labels.h5py')],
            'verbose':[1],
            #'decay':[10e-6],
            #'momentum':[0],
            #'noise':[0.01],
            #'reg_rate':[0.01,0.001],
            #'learning_rate':[10e-3, 10e-4] #np.logspace(-6,-3, 4)
        }

        p = {}
        for key in params_range:
            p[key] = np.random.choice(params_range[key])

        with open(CONF_PATH, 'wb') as output:
            pickle.dump(p, output, pickle.HIGHEST_PROTOCOL)


    def test_train_model_from_conf(self, indices):
        CONF_PATH = os.path.join(PARAMETERS_DIR,os.environ['SGE_TASK_ID'])

        with open(CONF_PATH, 'rb') as input:
            p = pickle.load(input)
            print("Using params {}".format(p))
            history, model = create_dense_model(indices.train, indices.test, p)
            print("Number of weights: {}".format(model.count_params()))
            assert(np.max(history.history['val_acc']) > 0.70)
            model_name = generate_name_from_params(p)
            model.save(os.path.join(TEST_DIR,'exported_models',model_name))

    
    


