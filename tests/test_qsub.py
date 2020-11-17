import os

import keras

from helpers import  generate_name_from_params


import numpy as np
import pickle

from parameters_complete import (TEST_DIR, SYN_DATA_DIR, PARAMETERS_DIR)

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
            'feature_matrix_path': [os.path.join(SYN_DATA_DIR, '3d_feature_matrix.npy')],
            'y_path':[os.path.join(SYN_DATA_DIR, 'syn_labels.h5py')],
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


    
