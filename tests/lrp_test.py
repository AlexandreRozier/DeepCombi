
import math
import os
import innvestigate
import innvestigate.utils as iutils
import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from keras.utils import plot_model
from keras import callbacks
from keras.layers import Dense, Dropout, Conv1D, Flatten
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import talos
from talos import Reporting, Evaluate, Deploy, Restore
from combi import combi_method
from helpers import chi_square, permuted_combi, string_to_featmat   
from parameters_complete import (Cs, classy, filter_window_size,
                                 p_pnorm_filter, pnorm_feature_scaling,
                                 svm_rep)
from models import create_conv_model, create_dense_model


@pytest.mark.incremental
class TestLrp(object):


    def test_talos_training(self, feature_matrix, labels, talos_output_dir, test_dir):
        
        
        # Preprocess data

        p = {
            'epochs': [50],
            'dropout_rate': (0.1,0.7, 1),
            'batch_size': [32]
        }
    
        s = talos.Scan(
            feature_matrix.train, 
            labels.train, 
            model=create_conv_model, 
            params = p, 
            dataset_name= talos_output_dir + "syn_wtcc", 
            x_val=feature_matrix.test,
            y_val=labels.test,
            #grid_downsample=0.1,
            experiment_no='linear')

        # K-fold cross validation on validation set 
        x_val = np.expand_dims(feature_matrix.val, axis=2) 
        l_val = np.expand_dims(labels.val, axis=2) 
        val_acc_scores = Evaluate(s).evaluate(x_val, l_val, print_out=True)
        assert(np.mean(val_acc_scores) > 0.50)

        
        
        

    def test_with_optimal_params(self, feature_matrix, talos_output_dir, labels, test_dir):

        r = Reporting(talos_output_dir+'syn_wtcc_linear.csv')
        bp = r.best_params('val_acc')[0]
        p = {
            'epochs': int(bp[0]),
            'dropout_rate': bp[1],
            'batch_size': int(bp[2])
        }
        
        history, model = create_conv_model(feature_matrix.train, labels.train, feature_matrix.test, labels.test, p)

        #np.expand_dims(x_train, axis=2) 
        #x_test = np.expand_dims(x_test, axis=2) 
        loss, val_acc = model.evaluate(feature_matrix.val, labels.val)
        print("Params: {}; Loss: {}; val_accuracy: {}".format(bp, loss, val_acc))
        model.save(test_dir+'/exported_models/linear.h5')
        assert val_acc > 0.70




    @pytest.mark.run(after='test_with_optimal_params')
    def test_lrp_innvestigate_analysis(self, test_dir, feature_matrix):

        model = load_model(test_dir+'/exported_models/linear.h5')
        model.get_layer(name='dense_1').name = 'dense_first'

        # Stripping the softmax activation from the model
        model_wo_sm = iutils.keras.graph.model_wo_softmax(model)

        # Creating an analyzer
        dtaylor_analyzer = innvestigate.analyzer.LRPZPlus(model_wo_sm)

        n_individuals, features_dim = feature_matrix.raw.shape
        #feature_matrix = np.expand_dims(feature_matrix, axis=2) 

        analysis = dtaylor_analyzer.analyze(feature_matrix.raw)
        analysis = analysis.reshape(n_individuals, -1, 3).sum(axis=2)
        x = range(4800, 5200)
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        #for i in range(n_individuals):
        ax1.scatter(x, analysis.sum(0), marker='x', alpha=0.7)
        """
        for i in range(4):
            ax1 = fig.add_subplot(2,2,i+1)
            for j in range(3):
                indices = np.argpartition(analysis[3*i+j], -30)[-30:]
                ax1.scatter(indices, analysis[3*i+j][indices], marker='x',alpha=0.7)
            """
        plt.show()
        print("done")


