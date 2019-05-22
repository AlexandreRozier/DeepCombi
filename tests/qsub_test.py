
import math
import os
import keras
import numpy as np
import pytest
from tqdm import tqdm
import pandas as pd
import pickle
import time
import keras.constraints
from sklearn.model_selection import train_test_split, KFold
from helpers import chi_square, string_to_featmat, EnforceNeg 
keras.constraints.EnforceNeg = EnforceNeg # Absolutely crucial

from parameters_complete import (Cs, classy, filter_window_size,
                                 p_pnorm_filter, pnorm_feature_scaling,
                                 svm_rep, TEST_DIR, TALOS_OUTPUT_DIR, DATA_DIR, PARAMETERS_DIR)
from models import create_conv_model, create_dense_model

@pytest.mark.incremental
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
            'epochs': np.linspace(100,300, 30),
            'dropout_rate': np.linspace(0.3,0.5,2),
            'batch_size': np.linspace(128, 640, 10),
            'feature_matrix_path': [os.path.join(DATA_DIR,'3d_feature_matrix.npy')],
            'y_path':[os.path.join(DATA_DIR,'syn_labels.txt')],
            'verbose':[1],
            'decay':[0],
            'momentum':[0],
            'learning_rate':np.logspace(-4,-2, 3)
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
            history, model = create_conv_model(indices.train, indices.test, p)
            assert(history.history['val_acc'][-1] > 0.50)
    
    


