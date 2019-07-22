import numpy as np
import pandas as pd
import os 
import time

import tensorflow


import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import SGD
from models import create_explanable_conv_model, create_explanable_conv_model, create_lenet_model

from tqdm import tqdm
from combi import combi_method
from helpers import postprocess_weights, chi_square, compute_metrics, plot_pvalues, generate_name_from_params

from parameters_complete import random_state, nb_of_nodes, pnorm_feature_scaling, filter_window_size, p_svm, p_pnorm_filter, n_total_snps, top_k, ttbr, thresholds, IMG_DIR
from joblib import Parallel, delayed
from combi import classifier

import innvestigate


best_params = {
    'epochs': 6,
    'batch_size': 32,
    'l1_reg': 1e-6,
    'l2_reg': 1e-5,
    'dropout_rate': 0.3,
    'kernel_window_size': 30,
    'filter_nb':5,
    'optimizer': 'adam',
    'patience': 7
}

class TestCNN(object):

    def test_train(self, fm, labels_0based, indices):

        x = fm('2d')['0'][:]
        x = np.expand_dims(x, axis=-1)
        y = labels_0based['0']
        model = create_explanable_conv_model(best_params)
        model.fit(x=x[indices.train],
                y=y[indices.train],
                validation_data=(x[indices.test], y[indices.test]),
                epochs=best_params['epochs'],
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=best_params['patience'],
                                    mode='min', verbose=1),
                ],
            verbose=1)


    def test_conv_lrp(self, h5py_data, labels, fm, labels_0based, indices, tmp_path):
        
        features = fm("3d")['0'][:]
        labels_0b = labels_0based['0']
        labels_ = labels['0']
        model_wo_sm = create_explanable_conv_model(best_params)
        model_wo_sm.fit(x=features[indices.train],
                  y=labels_0b[indices.train],
                  validation_data=(features[indices.test], labels_0b[indices.test]),
                  epochs=best_params['epochs'],
                  callbacks=[
                    EarlyStopping(monitor='val_loss', patience=best_params['patience'], mode='min', verbose=1)
        ],
            verbose=1)
        
        complete_pvalues = chi_square(h5py_data['0'][:], labels_)
        fig, axes = plt.subplots(4,2, squeeze=True)
        
        # LRPZ
        analyzer = innvestigate.analyzer.LRPZ(model_wo_sm)
        weights = analyzer.analyze(features[labels_0b>0.9]).sum(0)
        top_indices_sorted = postprocess_weights(
            weights, top_k, filter_window_size, p_svm, p_pnorm_filter)
        plot_pvalues(complete_pvalues, top_indices_sorted, axes[0][0])
        axes[0][0].legend(["LRPZ, 300 subjects, dense; ttbr={}".format(ttbr)])
        axes[0][1].plot(np.absolute(weights).reshape(-1,3).sum(1), label='ttbr={}'.format(ttbr))

        
        # DeepTaylor
        analyzer = innvestigate.analyzer.DeepTaylor(model_wo_sm)
        weights = analyzer.analyze(features[labels_0b>0.9]).sum(0)
        top_indices_sorted = postprocess_weights(
            weights, top_k, filter_window_size, p_svm, p_pnorm_filter)
        plot_pvalues(complete_pvalues, top_indices_sorted, axes[1][0])
        axes[1][0].legend(["Deep Taylor, 300 subjects, dense; ttbr={}".format(ttbr)])
        axes[1][1].plot(np.absolute(weights).reshape(-1,3).sum(1), label='ttbr={}'.format(ttbr))

        # LRPAlpha1Beta0
        analyzer = innvestigate.analyzer.LRPAlpha1Beta0(model_wo_sm)
        weights = analyzer.analyze(features[labels_0b>0.9]).sum(0)
        top_indices_sorted = postprocess_weights(
            weights, top_k, filter_window_size, p_svm, p_pnorm_filter)
        plot_pvalues(complete_pvalues, top_indices_sorted, axes[2][0])
        axes[2][0].legend(["LRPAlpha1Beta0, 300 subjects, dense; ttbr={}".format(ttbr)])
        axes[2][1].plot(np.absolute(weights).reshape(-1,3).sum(1), label='ttbr={}'.format(ttbr))


        # LRPAlpha2Beta1
        analyzer = innvestigate.analyzer.LRPAlpha2Beta1(model_wo_sm)
        weights = analyzer.analyze(features[labels_0b>0.9]).sum(0)
        top_indices_sorted = postprocess_weights(
            weights, top_k, filter_window_size, p_svm, p_pnorm_filter)
        plot_pvalues(complete_pvalues, top_indices_sorted, axes[3][0])
        axes[3][0].legend(["LRPAlpha2Beta1, 300 subjects, dense; ttbr={}".format(ttbr)])
        axes[3][1].plot(np.absolute(weights).reshape(-1,3).sum(1), label='ttbr={}'.format(ttbr))

        fig.savefig(os.path.join(IMG_DIR, 'conv-lrp.png'))


    def test_hp_params(self, fm, labels_0based, indices, rep, output_path):
        fm = fm('2d')
        
        datasets = [np.expand_dims(fm[str(i)][:], axis=-1) for i in range(rep)]

        params_space = {
            'epochs': [500],
            'batch_size': [32],
            'l1_reg': np.logspace(-7, -3, 5),
            'l2_reg': np.logspace(-7, -3, 5),
            #'dropout_rate': np.linspace(0.1, 0.5, 5),
            'kernel_window_size': 3*[30,35],
            #'filter_nb':[5,10,20,30],
            'optimizer': ['adam'], 
        }

        BUDGET = 100
        grid = ParameterGrid(params_space)
        grid = random_state.choice(list(grid), BUDGET)
        
        print("TESTING {} PARAMETERS".format(len(grid)))
        params_array_per_node = np.array_split(grid, nb_of_nodes)

        def f(g):

            time.sleep(0.1)
            name = generate_name_from_params(g)
            with tensorflow.Session().as_default():
                model = create_lenet_model(g)

                histories = [model.fit(x=fm[indices[str(i)].train],
                                        y=labels_0based[str(i)][indices[str(i)].train],
                                        validation_data=(
                                           fm[indices[str(i)].test], labels_0based[str(i)][indices[str(i)].test]),
                                        epochs=g['epochs'],
                                        callbacks=[
                                            EarlyStopping(monitor='val_loss', patience=15, mode='min'),
                                            TensorBoard(log_dir=os.path.join(output_path,'tb',name+str(i)),
                                                histogram_freq=3,
                                                write_graph=False,
                                                write_grads=True,
                                                write_images=False,
                                            )
                                        ],
                                        verbose=1).history for i, fm in enumerate(datasets)]
                mean_val_loss = np.mean([history['val_loss'][-1] for history in histories])
                for history in histories:
                    history['mean_val_loss'] = mean_val_loss

            return [{**g, **history} for history in histories]

        hparams_array = params_array_per_node[int(os.environ['SGE_TASK_ID'])-1]

        entries = np.array(Parallel(n_jobs=-1, prefer="threads")
                           (delayed(f)(g) for g in hparams_array)).flatten()
        results = pd.DataFrame(list(entries))
        results.to_csv(os.path.join(output_path, os.environ['SGE_TASK_ID']+'.csv'))


    def test_svm_cnn_comparison(self, fm, labels, labels_0based, rep, indices):
        """ Compares performance of SVM and CNN models
        """
        
        fm_2d = fm("2d")

        # 1. Train cnn with optimal params

        def fit_cnn(x, y):
            x = np.expand_dims(x, axis=-1)
            montaez_model = create_explanable_conv_model(best_params)
            return montaez_model.fit(x=x[indices.train],
                                     y=y[indices.train],
                                     validation_data=(
                                         x[indices.test], y[indices.test]),
                                     epochs=best_params['epochs'],
                                     callbacks=[
                                        EarlyStopping(monitor='val_loss', patience=best_params['patience'], mode='min'),
                                    ]).history['val_acc'][-1]

        def fit_svm(x, y):
            svm_model = classifier.fit(x[indices.train], y[indices.train])
            return svm_model.score(x[indices.test], y[indices.test])

        cnn_val_acc = Parallel(
            n_jobs=-1)(delayed(fit_cnn)(fm_2d[str(i)][:], labels_0based[str(i)]) for i in tqdm(range(1, rep)))
        svm_val_acc = Parallel(
            n_jobs=-1)(delayed(fit_svm)(fm_2d[str(i)][:], labels[str(i)]) for i in tqdm(range(1, rep)))

        cnn_val_acc = cnn_val_acc * 100
        svm_val_acc = svm_val_acc * 100

        # 3. Compare average val_acc on all syn datasets
        print('SVM val_acc mean {}/std{}/min{}/max{}; cnn val_acc mean {}/std{}/max{}/min{}'.format(
            np.mean(svm_val_acc),np.std(svm_val_acc), np.max(svm_val_acc),np.min(svm_val_acc),
            np.mean(cnn_val_acc),np.std(cnn_val_acc), np.max(cnn_val_acc),np.min(cnn_val_acc)))




    def test_tpr_fwer(self, h5py_data, labels, labels_0based, fm, indices, rep, true_pvalues):
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

                model_wo_sm = create_explanable_conv_model(best_params)

                model_wo_sm.fit(x=x[indices.train],
                                        y=l_0b[indices.train],
                                        validation_data=(
                                            x[indices.test], l_0b[indices.test]),
                                        epochs=best_params['epochs'],
                                        callbacks=[
                                                EarlyStopping(monitor='val_loss', patience=best_params['patience'], mode='min'),
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
                 label='Conv method - ttbr={}'.format(ttbr))

        ax2.plot(tpr_combi, precision_combi, '-o',
                 label='Combi method - ttbr={}'.format(ttbr))
        ax2.plot(tpr_dense, precision_dense, '-x',
                 label='Conv method - ttbr={}'.format(ttbr))

        ax1.legend()
        ax2.legend()
        fig.savefig(os.path.join(
            IMG_DIR, 'tpr_fwer_conv_combi_comparison.png'), dpi=300)
