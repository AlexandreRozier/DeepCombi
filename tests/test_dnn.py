
import copy
import h5py
import tensorflow
from keras.utils import to_categorical
import innvestigate
from combi import classifier
from torch.utils.data.sampler import SubsetRandomSampler
import torch.multiprocessing as mp
from combi import combi_method
from helpers import EnforceNeg, generate_name_from_params, chi_square, postprocess_weights, compute_metrics, plot_pvalues, generate_syn_phenotypes, train_torch_model, evaluate_torch_model
from models import DataGenerator, train_dummy_dense_model, create_montaez_dense_model, MontaezEarlyStopping, best_params_montaez
from parameters_complete import (Cs, IMG_DIR, PARAMETERS_DIR, DATA_DIR, SAVED_MODELS_DIR,
                                 TEST_DIR, n_total_snps, seed, p_pnorm_filter, filter_window_size, pnorm_feature_scaling, p_svm, noise_snps, inform_snps, top_k,  thresholds,
                                 random_state, nb_of_nodes)
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l1
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Flatten, Activation
from keras.models import Sequential, load_model
from keras.optimizers import SGD
import keras
from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import time
import torch
from Indices import Indices
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')



class TestDNN(object):

    def test_dense_model(self, h5py_data, labels,ttbr, fm, labels_0based, indices, tmp_path):
      
        
        fm_ = fm("3d")

        features = fm_['0'][:]
        labels_0b = labels_0based['0']
        labels_ = labels['0']

        model_wo_sm = create_montaez_dense_model(best_params_montaez)
        model_wo_sm.fit(x=features[indices.train],
                  y=labels_0b[indices.train],
                  validation_data=(features[indices.test], labels_0b[indices.test]),
                  epochs=best_params_montaez['epochs'],
                  callbacks=[
                    EarlyStopping(monitor='val_loss', patience=best_params_montaez['patience'], mode='min', verbose=1)
        ],
            verbose=1)
        

        analyzer = innvestigate.analyzer.LRPZ(model_wo_sm)
        weights = analyzer.analyze(features).sum(0)

        top_indices_sorted = postprocess_weights(
            weights, top_k, filter_window_size, p_svm, p_pnorm_filter)

        complete_pvalues = chi_square(h5py_data['0'][:], labels_)

        fig, axes = plt.subplots(1, squeeze=True)

        plot_pvalues(complete_pvalues, top_indices_sorted, axes)
        axes.legend(["LRPZ, 300 subjects, dense; ttbr={}".format(ttbr)])
        fig.savefig(os.path.join(IMG_DIR, 'dense-lrp.png'))

    def test_train_montaez(self, fm, labels_0based, indices):
        g = {
            'epochs': 400,
            'batch_size': 32,
            'l1_reg': 3 * 1e-4,
            'l2_reg': 1e-6,
            'dropout_rate': 0.3,
            'optimizer':  SGD(lr=0.001)
        }

        x = fm('3d')['0'][:]
        x = np.expand_dims(x, axis=2)
        y = labels_0based['0']
        model = create_montaez_dense_model(g)
        model.fit(x=x[indices.train],
                  y=y[indices.train],
                  validation_data=(x[indices.test], y[indices.test]),
                  epochs=g['epochs'],
                  callbacks=[
            MontaezEarlyStopping(monitor='val_loss', patience=20,
                          mode='min', verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10, mode='min', verbose=1)
        ],
            verbose=1)

    def test_plot_dense(self, h5py_data, fm, indices, tmp_path):
        """ Compares efficiency of the combi method with several TTBR
        """
        ttbrs = [20, 6, 1, 0]
        h5py_data = h5py_data['0'][:]

        fig, axes = plt.subplots(len(ttbrs), 3, sharex='col')
        x_3d = fm("3d")['0'][:]
        x_2d = fm("2d")['0'][:]

        for i, ttbr in enumerate(ttbrs):
            print('Using tbrr={}'.format(ttbr))
            labels = generate_syn_phenotypes(
                root_path=DATA_DIR, ttbr=ttbr)['0']
            l_0b = (labels+1)/2
           
            model_wo_sm = create_montaez_dense_model(best_params_montaez)
            model_wo_sm.fit(x=x_3d[indices.train],
                  y=l_0b[indices.train],
                  validation_data=(x_3d[indices.test], l_0b[indices.test]),
                  epochs=best_params_montaez['epochs'],
                  callbacks=[
                    EarlyStopping(monitor='val_loss', patience=best_params_montaez['patience'], mode='min', verbose=1),
                ],
            )

            analyzer = innvestigate.analyzer.LRPZ(model_wo_sm)
            weights = analyzer.analyze(x_3d[l_0b>0.9]).sum(0) # sum over sick ppl

            top_indices_sorted = postprocess_weights(
                weights, top_k, filter_window_size, p_svm, p_pnorm_filter)

            complete_pvalues = chi_square(h5py_data, labels)
            plot_pvalues(complete_pvalues, top_indices_sorted, axes[i][0])
            
            # Plot distribution of relevance
            axes[i][1].plot(np.absolute(weights).reshape(-1,3).sum(1), label='ttbr={}'.format(ttbr))
            axes[i][1].legend()
            axes[i][1].set_title('Absolute relevance distribution')

            # Plot distribution of svm weights
            svm_weights = classifier.fit(x_2d, labels).coef_
            axes[i][2].plot(np.absolute(svm_weights).reshape(-1,3).sum(1), label='ttbr={}'.format(ttbr))
            axes[i][2].legend()
            axes[i][2].set_title('Absolute SVM weight distribution')

            fig.savefig(os.path.join(IMG_DIR, 'manhattan-dense-test.png'))

    def test_hp_params(self, fm, labels_0based, indices, rep, output_path):
        fm = fm('3d')
        datasets = [fm[str(i)][:] for i in range(rep)]

        params_space = {
            'epochs': [500],
            'batch_size': [32],
            'l1_reg': np.logspace(-6, -1, 5),
            'l2_reg': np.logspace(-6, -1, 5),
            'dropout_rate': np.linspace(0.1, 0.5, 5),
            'optimizer': ['adam'], #[dict(lr=rate, decay=0) for rate in [1e-5, 1e-4, 1e-3]]
        }

        grid = ParameterGrid(params_space)
        grid = [params for params in grid]
        params_array_per_node = np.array_split(grid, nb_of_nodes)

        def f(g):
            time.sleep(0.1)
            with tensorflow.Session().as_default():
                model = create_montaez_dense_model(g)

                histories = [model.fit(x=fm[indices.train],
                                       y=labels_0based[str(i)][indices.train],
                                       validation_data=(
                                           fm[indices.test], labels_0based[str(i)][indices.test]),
                                       epochs=g['epochs'],
                                       callbacks=[
                                            EarlyStopping(monitor='val_loss', patience=7, mode='min'),
                                        ],
                                        verbose=1).history for i, fm in enumerate(datasets)]

            return [{**g, **history} for history in histories]

        hparams_array = params_array_per_node[int(os.environ['SGE_TASK_ID'])-1]

        entries = np.array(Parallel(n_jobs=-1, prefer="threads")
                           (delayed(f)(g) for g in hparams_array)).flatten()
        results = pd.DataFrame(list(entries))
        results.to_csv(output_path)

    def test_tpr_fwer(self, h5py_data, labels, ttbr, labels_0based, fm, indices, rep, true_pvalues):
        """ Compares combi vs dense curves
        """

        fig, axes = plt.subplots(2)
        fig.set_size_inches(18.5, 10.5)
        ax1, ax2 = axes

        ax1.set_ylabel('TPR')
        ax1.set_xlabel('FWER')
        ax1.set_ylim(0, 0.45)
        ax1.set_xlim(0, 0.1)

        ax2.set_ylabel('Precision')
        ax2.set_xlabel('TPR')

        def combi_compute_pvalues(d, x, l):

            indices, pvalues = combi_method(d, x, l, pnorm_feature_scaling,
                                            filter_window_size, top_k)
            pvalues_filled = np.ones(n_total_snps)
            pvalues_filled[indices] = pvalues
            del d, l
            return pvalues_filled

        def dense_compute_pvalues(d, x, l_0b, l):

            time.sleep(0.01)
            with tensorflow.Session().as_default():

                model_wo_sm = create_montaez_dense_model(best_params_montaez)

                model_wo_sm.fit(x=x[indices.train],
                                        y=l_0b[indices.train],
                                        validation_data=(
                                            x[indices.test], l_0b[indices.test]),
                                        epochs=best_params_montaez['epochs'],
                                        callbacks=[
                                                EarlyStopping(monitor='val_loss', patience=best_params_montaez['patience'], mode='min'),
                                            ])

            

                analyzer = innvestigate.analyzer.LRPZ(model_wo_sm)
                weights = analyzer.analyze(x[l_0b>0.9]).sum(0)

                top_indices_sorted = postprocess_weights(
                    weights, top_k, filter_window_size, p_svm, p_pnorm_filter)
                pvalues = chi_square(d[:, top_indices_sorted], l)
                pvalues_filled = np.ones(n_total_snps)
                pvalues_filled[top_indices_sorted] = pvalues
                del d, x, l

            return pvalues_filled

        fm_2d = fm("2d")
        fm_3d = fm("3d")

        pvalues_per_run_combi = Parallel(n_jobs=-1, require='sharedmem')(delayed(
            combi_compute_pvalues)(h5py_data[str(i)][:], fm_2d[str(i)][:], labels[str(i)]) for i in tqdm(range(rep)))

        pvalues_per_run_dense = Parallel(n_jobs=-1, require='sharedmem')(delayed(
            dense_compute_pvalues)(h5py_data[str(i)][:], fm_3d[str(i)][:], labels_0based[str(i)], labels[str(i)]) for i in tqdm(range(rep)))

        res_combi = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(
            pvalues_per_run_combi, true_pvalues, rep, threshold) for threshold in tqdm(thresholds)))

        res_dense = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(
            pvalues_per_run_dense, true_pvalues, rep, threshold) for threshold in tqdm(thresholds)))

        tpr_combi, enfr_combi, fwer_combi, precision_combi = res_combi.T
        tpr_dense, enfr_dense, fwer_dense, precision_dense = res_dense.T

        assert fwer_combi.max() <= 1 and fwer_combi.min() >= 0
        ax1.plot(fwer_combi, tpr_combi, '-o',
                 label='Combi method - ttbr={}'.format(ttbr))
        ax1.plot(fwer_dense, tpr_dense, '-x',
                 label='Dense method - ttbr={}'.format(ttbr))

        ax2.plot(tpr_combi, precision_combi, '-o',
                 label='Combi method - ttbr={}'.format(ttbr))
        ax2.plot(tpr_dense, precision_dense, '-x',
                 label='Dense method - ttbr={}'.format(ttbr))

        ax1.legend()
        ax2.legend()
        fig.savefig(os.path.join(
            IMG_DIR, 'tpr_fwer_dense_combi_comparison_ttbr_{}.png'.format(ttbr)), dpi=300)

    def test_svm_dnn_comparison(self, fm, labels, labels_0based, rep, indices):
        """ Compares performance of SVM and DNN models
        """
        
        fm_2d = fm("2d")
        fm_3d = fm("3d")

        # 1. Train dnn with optimal params

        def fit_dnn(x, y):
            montaez_model = create_montaez_dense_model(best_params_montaez)
            return montaez_model.fit(x=x[indices.train],
                                     y=y[indices.train],
                                     validation_data=(
                                         x[indices.test], y[indices.test]),
                                     epochs=best_params_montaez['epochs'],
                                     callbacks=[
                                        EarlyStopping(monitor='val_loss', patience=best_params_montaez['patience'], mode='min', verbose=1),
                                    ],
                                    verbose=0).history['val_acc'][-1]

        def fit_svm(x, y):
            svm_model = classifier.fit(x[indices.train], y[indices.train])
            return svm_model.score(x[indices.test], y[indices.test])

        dnn_val_acc = Parallel(
            n_jobs=-1)(delayed(fit_dnn)(fm_3d[str(i)][:], labels_0based[str(i)]) for i in range(1, rep))
        svm_val_acc = Parallel(
            n_jobs=-1)(delayed(fit_svm)(fm_2d[str(i)][:], labels[str(i)]) for i in range(1, rep))

        # 3. Compare average val_acc on all syn datasets
        print('SVM mean / std val acc:{}+-{}; DNN mean / std val acc:{}+-{}'.format(
            np.mean(svm_val_acc),np.std(svm_val_acc), np.mean(dnn_val_acc),np.std(dnn_val_acc)))
