
import math
import os
import innvestigate
import innvestigate.utils as iutils
import keras
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pytest
from tqdm import tqdm

from keras.utils import plot_model
from keras import callbacks
from keras.layers import Dense, Dropout, Conv1D, Flatten
from keras.models import Sequential, load_model, model_from_json, Model
from keras.utils import to_categorical
import keras.constraints
from sklearn.model_selection import train_test_split, KFold
import talos
from talos import Reporting, Evaluate, Deploy, Restore
from helpers import chi_square, string_to_featmat, EnforceNeg 
keras.constraints.EnforceNeg = EnforceNeg # Absolutely crucial

from parameters_complete import (Cs, classy, filter_window_size,
                                 p_pnorm_filter, pnorm_feature_scaling,
                                 svm_rep, TEST_DIR, TALOS_OUTPUT_DIR, DATA_DIR)
from models import create_conv_model, create_dense_model
from tests.conftest import seed

@pytest.mark.incremental
class TestLrp(object):
    

    def test_talos_hpopt(self, indices):
        """Performs Hyperparameters Optimization
        """  

        p = {
            'epochs': (1,10,10),
            'dropout_rate': (0.1,0.5, 3),
            'batch_size': [32,64],
            'feature_matrix_path': [os.path.join(DATA_DIR,'3d_feature_matrix.npy')],
            'y_path':[os.path.join(DATA_DIR,'syn_labels.txt')],
            'verbose':[1],
            'use_normalization':[True, False],
        }
    
        
        s = talos.Scan(
            indices.train, 
            indices.train, 
            model=create_conv_model, 
            params = p, 
            dataset_name= os.path.join(TALOS_OUTPUT_DIR, "syn_wtcc"), 
            x_val=indices.test,
            y_val=indices.test,
            grid_downsample=0.1,
            random_method='uniform_mersenne',
            experiment_no='conv',
            seed=seed)

        # K Fold evaluation on validation set for the best model
        #scores = Evaluate(s).evaluate(x_val_indexes,y_val_indexes, folds = 3, metric='val_acc', print_out=True)
        #assert np.mean(scores) > 0.7
        #assert np.mean(scores) < 1.0
        

    def test_with_optimal_params(self, indices):
        """ Trains and validate through KFold a model with the optimal hyperparams previously exhumed
        """

        r = Reporting(os.path.join(TALOS_OUTPUT_DIR,'syn_wtcc_conv.csv'))
        bp = r.best_params('val_acc')[0]
        print("Using hyperparameters: {}".format(bp))
        p = {
            'epochs': int(bp[0]),
            'batch_size': int(bp[1]),
            'dropout_rate':float(bp[2]),
            'feature_matrix_path': bp[3],
            'y_path':bp[4],
            'verbose':[1]
        }

        # K-fold cross validation on validation set 
        kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
        val_acc_scores = []
        for train, test in tqdm(kfold.split(indices.val)):
            print("Runnin K-Fold C-V on: {} Train, {} Test".format(len(train), len(test)))
            history, model = create_conv_model(indices.val[train], indices.val[train], indices.val[test], indices.val[test], p)
            val_acc_scores.append(history.history['val_acc'])
        print("Params: {}; Mean Accuracy: {}; Std: {}".format(bp, np.mean(val_acc_scores), np.std(val_acc_scores)))
        
        assert(np.mean(val_acc_scores) > 0.70)
        model.save(os.path.join(TEST_DIR,'exported_models','conv.h5'))


    # def test_save_model(self, f_and_l, indices, tmp_path):
    #     """ Tests if model saves correctly with custom Constraints
    #     """

    #     r = Reporting(os.path.join(TALOS_OUTPUT_DIR,'syn_wtcc_conv.csv'))
    #     bp = r.best_params('val_acc')[0]
    #     p = {
    #         'epochs': int(bp[0]),
    #         'batch_size': bp[4],
    #         'feature_matrix_path': os.path.join(DATA_DIR,'3d_feature_matrix.npy'),
    #         'y_path':os.path.join(DATA_DIR,'syn_labels.txt'),
    #         'verbose':[1]
    #     }
    #     history, model = create_conv_model(indices.train, indices.train, indices.test, indices.test, p)
    #     model.save(os.path.join(tmp_path,'conv.h5'))
    #     model2 = keras.models.load_model(os.path.join(tmp_path,'conv.h5'),
    #           custom_objects={'EnforceNeg':EnforceNeg})
 
    #     predictions = model2.predict(f_and_l['features'][indices.val][:2])
    #     ground_truth = f_and_l['labels'][indices.val][:2]
    #     np.testing.assert_allclose(predictions, ground_truth, atol=1e-01)



    @pytest.mark.run(after='test_with_optimal_params')
    def test_lrp_innvestigate_analysis(self, f_and_l):
        """ Tests the LRP result on our best model
        """
        features = f_and_l['features']
        model = load_model(os.path.join(TEST_DIR,'exported_models','conv.h5'), custom_objects={'EnforceNeg':EnforceNeg})
        model_wo_sm = iutils.keras.graph.model_wo_softmax(model)
       
        # Creating an analyzer

        n_individuals, features_dim, channels = features.all.shape
        
        inf = 0
        sup = 10020
        x = range(inf, sup)
        f, (ax1,ax2,ax3, ax4) = plt.subplots(4)
        dtaylor_analyzer = innvestigate.analyzer.LRPZ(model_wo_sm)
        analysis = dtaylor_analyzer.analyze(features.all).sum(axis=2)
        analysis = analysis[:,inf:sup]
        ax1.scatter(x, analysis.sum(0), marker='x', alpha=0.7, label="LRPZ")
        ax1.axvspan(5001, 5020, alpha=0.5, color='red')
        ax1.legend(loc="upper right")

        dtaylor_analyzer = innvestigate.analyzer.LRPZPlus(model_wo_sm)
        analysis = dtaylor_analyzer.analyze(features.all).sum(axis=2)
        analysis = analysis[:,inf:sup]
        ax2.scatter(x, analysis.sum(0), marker='x', alpha=0.7, label='LRPZPlus')
        ax2.axvspan(5001, 5020, alpha=0.5, color='red')
        ax2.legend(loc="upper right")

        dtaylor_analyzer = innvestigate.analyzer.DeepTaylor(model_wo_sm, {'low':-1,'high':1})
        analysis = dtaylor_analyzer.analyze(features.all).sum(axis=2)
        analysis = analysis[:,inf:sup]
        ax3.scatter(x, analysis.sum(0), marker='x', alpha=0.7, label='DeepTaylor')
        ax3.axvspan(5001, 5020, alpha=0.5, color='red')
        ax3.legend(loc="upper right")

        dtaylor_analyzer = innvestigate.analyzer.Gradient(model_wo_sm)
        analysis = dtaylor_analyzer.analyze(features.all).sum(axis=2)
        analysis = analysis[:,inf:sup]
        ax4.scatter(x, analysis.sum(0), marker='x', alpha=0.7, label='Gradient')
        ax4.axvspan(5001, 5020, alpha=0.5, color='red')
        ax4.legend(loc="upper right")

        #for i in range(n_individuals):
        """
        for i in range(4):
            ax1 = fig.add_subplot(2,2,i+1)
            for j in range(3):
                indices = np.argpartition(analysis[3*i+j], -30)[-30:]
                ax1.scatter(indices, analysis[3*i+j][indices], marker='x',alpha=0.7)
        """
        plt.show()


