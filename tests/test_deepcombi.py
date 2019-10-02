import numpy as np
import pandas as pd
import os
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow
from sklearn.model_selection import ParameterGrid

from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.optimizers import SGD
from models import create_explanable_conv_model, create_explanable_conv_model, create_lenet_model, \
    create_clairvoyante_model, create_montaez_dense_model, create_montaez_dense_model_2, create_convdense_model
from models import ConvDenseRLRonP, best_params_montaez_2
from tqdm import tqdm
from combi import combi_method
from helpers import postprocess_weights, chi_square, compute_metrics, plot_pvalues, generate_name_from_params, \
    generate_syn_phenotypes

from parameters_complete import random_state, nb_of_jobs, pnorm_feature_scaling, filter_window_size, p_svm, \
    p_pnorm_filter, n_total_snps, top_k, ttbr, thresholds, IMG_DIR, DATA_DIR, NUMPY_ARRAYS, alpha_sig_toy
from joblib import Parallel, delayed
from combi import toy_classifier, permuted_deepcombi_method
import innvestigate
import innvestigate.utils as iutils



class TestDeepCOMBI(object):

    def test_indices(self, labels_0based, indices):
        idx = indices['0']
        test_labels = labels_0based['0'][idx.test]
        train_labels = labels_0based['0'][idx.train]
        print(test_labels)
        print(train_labels)
        print(len(test_labels[test_labels == 1]))
        print(len(test_labels[test_labels == 0]))
        print(len(train_labels[train_labels == 1]))
        print(len(train_labels[train_labels == 0]))

    def test_train(self, fm, labels_0based, indices, rep, output_path):

        fm_ = fm('3d')

        def f(x, y, idx, key):
            with tensorflow.Session().as_default():
                model = create_montaez_dense_model_2(best_params_montaez_2)
                model.fit(x=x[idx.train],
                          y=y[idx.train],
                          validation_data=(x[idx.test], y[idx.test]),
                          epochs=best_params_montaez_2['epochs'],
                          callbacks=[
                              ReduceLROnPlateau(monitor='val_loss',
                                                factor=best_params_montaez_2['factor'],
                                                patience=best_params_montaez_2['patience'],
                                                mode='min'),
                              TensorBoard(log_dir=os.path.join(output_path, 'tb', 'test' + key),
                                          histogram_freq=3,
                                          write_graph=False,
                                          write_grads=True,
                                          write_images=False,
                                          )
                          ],
                          verbose=1)

        Parallel(n_jobs=-1, prefer="threads")(delayed(f)(
            fm_[str(i)][:],
            labels_0based[str(i)],
            indices[str(i)],
            str(i)
        ) for i in range(rep))

    def test_conv_lrp(self, h5py_data, labels, fm, labels_0based, labels_cat, indices, rep, tmp_path):

        fig, axes = plt.subplots(6, 2 * rep, squeeze=True)
        fig.set_size_inches(30, 30)

        def f(i, x_3d, x_2d, y, y_0b, idx):
            with tensorflow.Session().as_default():
                model = create_montaez_dense_model_2(best_params_montaez_2)
                model.fit(x=x_3d[idx.train],
                          y=y_0b[idx.train],
                          validation_data=(x_3d[idx.test], y_0b[idx.test]),
                          epochs=best_params_montaez_2['epochs'],
                          callbacks=[

                              ReduceLROnPlateau(monitor='val_loss',
                                                factor=best_params_montaez_2['factor'],
                                                patience=best_params_montaez_2['patience'],
                                                mode='min'),
                          ],
                          verbose=1)
                model = iutils.keras.graph.model_wo_softmax(model)

                toy_classifier.fit(x_2d, y)
                svm_weights = toy_classifier.coef_[0]  # n_snps * 3
                axes[0][2 * i + 1].plot(np.absolute(svm_weights).reshape(-1, 3).sum(1))
                axes[1][2 * i + 1].plot(np.absolute(svm_weights).reshape(-1, 3).sum(1))
                axes[2][2 * i + 1].plot(np.absolute(svm_weights).reshape(-1, 3).sum(1))
                axes[3][2 * i + 1].plot(np.absolute(svm_weights).reshape(-1, 3).sum(1))
                axes[4][2 * i + 1].plot(np.absolute(svm_weights).reshape(-1, 3).sum(1))
                axes[5][2 * i + 1].plot(np.absolute(svm_weights).reshape(-1, 3).sum(1))

                # LRPZ
                analyzer = innvestigate.analyzer.LRPZ(model)
                weights = analyzer.analyze(x_3d).sum(0)
                axes[0][2 * i].plot(np.absolute(weights).reshape(-1, 3).sum(1), label='ttbr={},lrpz'.format(ttbr))

                # LRPEpsilon
                analyzer = innvestigate.analyzer.LRPEpsilon(model, epsilon=1e-5)
                weights = analyzer.analyze(x_3d).sum(0)
                axes[1][2 * i].plot(np.absolute(weights).reshape(-1, 3).sum(1), label='epsilon')

                # LRPAlpha1Beta0
                analyzer = innvestigate.analyzer.LRPAlpha1Beta0(model)
                weights = analyzer.analyze(x_3d).sum(0)
                axes[2][2 * i].plot(np.absolute(weights).reshape(-1, 3).sum(1), label='a-1, b-0')

                # LRPAlpha2Beta1
                analyzer = innvestigate.analyzer.LRPAlpha2Beta1(model)
                weights = analyzer.analyze(x_3d).sum(0)
                axes[3][2 * i].plot(np.absolute(weights).reshape(-1, 3).sum(1), label='a-2, b-1')

                # LRPZ+
                analyzer = innvestigate.analyzer.LRPZPlus(model)
                weights = analyzer.analyze(x_3d).sum(0)
                axes[4][2 * i].plot(np.absolute(weights).reshape(-1, 3).sum(1), label='lrpzplus')

                # LRPAlpha1Beta0IgnoreBias
                analyzer = innvestigate.analyzer.LRPAlpha1Beta0IgnoreBias(model)
                weights = analyzer.analyze(x_3d).sum(0)
                axes[5][2 * i].plot(np.absolute(weights).reshape(-1, 3).sum(1), label='LRPAlpha1Beta0IgnoreBias')

        Parallel(n_jobs=-1, prefer="threads")(
            delayed(f)(i, fm("3d")[str(i)][:], fm("2d")[str(i)][:], labels[str(i)], labels_cat[str(i)], indices[str(i)])
            for i in range(rep))

        fig.savefig(os.path.join(IMG_DIR, 'montaez-full-coca-lrp.png'))

    def test_hp_params(self, fm, labels_0based, labels_cat, indices, rep, output_path):
        fm = fm('3d')

        datasets = [fm[str(i)][:] for i in range(rep)]

        params_space = {
            'epochs': [1000],
            'patience': [10, 30, 50, 100],
            'factor': np.linspace(0.1, 0.8, 9),
            'dropout_rate': np.linspace(0.1, 0.5, 5),
            'l1_reg': np.logspace(-6, -2, 5),
            'l2_reg': np.logspace(-6, -2, 5),
            'lr': np.logspace(-4, -2, 3),
        }

        grid = ParameterGrid(params_space)
        BUDGET = 100
        grid = random_state.choice(list(grid), BUDGET)

        print("TESTING {} PARAMETERS".format(len(grid)))
        params_array_per_node = np.array_split(grid, nb_of_jobs)

        def f(g):
            name = generate_name_from_params(g)
            with tensorflow.Session().as_default():
                model = create_montaez_dense_model_2(g)

                histories = [model.fit(x=fm[indices[str(i)].train],
                                       y=labels_cat[str(i)][indices[str(i)].train],
                                       validation_data=(
                                           fm[indices[str(i)].test], labels_cat[str(i)][indices[str(i)].test]),
                                       epochs=g['epochs'],
                                       callbacks=[
                                           TensorBoard(log_dir=os.path.join(output_path, 'tb', name + str(i)),
                                                       histogram_freq=3,
                                                       write_graph=False,
                                                       write_grads=True,
                                                       write_images=False),
                                           ReduceLROnPlateau(monitor='val_loss',
                                                             factor=g['factor'],
                                                             patience=g['patience'],
                                                             mode='min'
                                                             ),
                                       ],
                                       verbose=1).history for i, fm in enumerate(datasets)]
                mean_val_loss = np.mean([history['val_loss'][-1] for history in histories])
                for history in histories:
                    history['mean_val_loss'] = mean_val_loss

            return [{**g, **history} for history in histories]

        hparams_array = params_array_per_node[int(os.environ['SGE_TASK_ID']) - 1]

        entries = np.array(Parallel(n_jobs=-1, prefer="threads")
                           (delayed(f)(g) for g in hparams_array)).flatten()
        results = pd.DataFrame(list(entries))
        results.to_csv(os.path.join(output_path, os.environ['SGE_TASK_ID'] + '.csv'))

    def test_svm_cnn_comparison(self, fm, labels, labels_cat, rep, indices):
        """ Compares performance of SVM and CNN models
        """

        fm_3d = fm("3d")
        fm_2d = fm("2d")

        def fit_cnn(x, y, idx):
            model = create_montaez_dense_model_2(best_params_montaez_2)
            return model.fit(x=x[idx.train],
                             y=y[idx.train],
                             validation_data=(
                                 x[idx.test], y[idx.test]),
                             epochs=best_params_montaez_2['epochs'],
                             callbacks=[
                                 ReduceLROnPlateau(monitor='val_loss',
                                                   factor=best_params_montaez_2['factor'],
                                                   patience=best_params_montaez_2['patience'],
                                                   mode='min'),
                             ]).history['val_acc'][-1]

        def fit_svm(x, y, idx):
            svm_model = toy_classifier.fit(x[idx.train], y[idx.train])
            return svm_model.score(x[idx.test], y[idx.test])

        cnn_val_acc = Parallel(
            n_jobs=30)(
            delayed(fit_cnn)(fm_3d[str(i)][:], labels_cat[str(i)], indices[str(i)]) for i in tqdm(range(1, rep)))
        svm_val_acc = Parallel(
            n_jobs=30)(delayed(fit_svm)(fm_2d[str(i)][:], labels[str(i)], indices[str(i)]) for i in tqdm(range(1, rep)))

        cnn_val_acc = cnn_val_acc * 100
        svm_val_acc = svm_val_acc * 100

        # 3. Compare average val_acc on all syn datasets
        print('SVM val_acc mean {}/std{}/max{}/min{}; cnn val_acc mean {}/std{}/max{}/min{}'.format(
            np.mean(svm_val_acc), np.std(svm_val_acc), np.max(svm_val_acc), np.min(svm_val_acc),
            np.mean(cnn_val_acc), np.std(cnn_val_acc), np.max(cnn_val_acc), np.min(cnn_val_acc)))

    def test_tpr_fwer(self, h5py_data, labels, labels_0based, labels_cat, fm, indices, rep, true_pvalues):
        """ Compares combi vs dense curves
        """

        window_lengths = [31, 35, 41]

        best_params_montaez_2['n_snps'] = n_total_snps
        n_permutations = 2

        def combi_compute_pvalues(d, x, l):

            idx, pvalues = combi_method(d, x, l, pnorm_feature_scaling,
                                        filter_window_size, top_k)
            pvalues_filled = np.ones(n_total_snps)
            pvalues_filled[idx] = pvalues
            del d, l
            return pvalues_filled

        def challenger_compute_pvalues(d, x, l_0b, l, idx):
            is_only_zeros = False
            with tensorflow.Session().as_default():

                model = create_montaez_dense_model_2(best_params_montaez_2)

                model.fit(x=x[idx.train], y=l_0b[idx.train],
                          validation_data=(x[idx.test], l_0b[idx.test]),
                          epochs=best_params_montaez_2['epochs'],
                          callbacks=[
                              ReduceLROnPlateau(monitor='val_loss',
                                                factor=best_params_montaez_2['factor'],
                                                patience=best_params_montaez_2['patience'],
                                                mode='min'),
                          ])

                model = iutils.keras.graph.model_wo_softmax(model)
                analyzer = innvestigate.analyzer.LRPAlpha1Beta0(model)
                weights = analyzer.analyze(x).sum(0)

                if np.max(abs(weights)) < 0.005:
                    fig, axes = plt.subplots(1)
                    is_only_zeros = True
                    axes.plot(np.absolute(weights).sum(axis=1))
                    fig.savefig(os.path.join(IMG_DIR, 'test.png'))

                pvalues_list = np.zeros((len(window_lengths), weights.shape[0]))
                for i, filter_size in enumerate(window_lengths):
                    top_indices_sorted, _ = postprocess_weights(
                        weights, top_k, filter_size, p_svm, p_pnorm_filter)
                    pvalues = chi_square(d[:, top_indices_sorted], l)
                    pvalues_filled = np.ones(n_total_snps)
                    pvalues_filled[top_indices_sorted] = pvalues
                    pvalues_list[i] = pvalues_filled
                del d, x, l

            return pvalues_list, is_only_zeros

        fm_2d = fm("2d")
        fm_3d = fm("3d")

        pvalues_per_run_combi = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(
            combi_compute_pvalues)(h5py_data[str(i)][:], fm_2d[str(i)][:], labels[str(i)]) for i in tqdm(range(rep))))

        pvalues_per_run_rpvt = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(
            chi_square)(h5py_data[str(i)][:], labels[str(i)]) for i in tqdm(range(rep))))

        # len(thresholds) * len(window_sizes) * 10020
        a = Parallel(n_jobs=-1, require='sharedmem')(delayed(
            challenger_compute_pvalues)(h5py_data[str(i)][:], fm_3d[str(i)][:], labels_cat[str(i)], labels[str(i)],
                                        indices[str(i)]) for i in tqdm(range(rep)))

        # INNvestigate bugfix
        zeros_index = np.array(list(np.array(a)[:, 1]))
        pvalues_per_run_dense = np.array(list(np.array(a)[:, 0]))

        pvalues_per_run_combi = pvalues_per_run_combi[np.logical_not(zeros_index)]
        pvalues_per_run_dense = pvalues_per_run_dense[np.logical_not(zeros_index)]
        pvalues_per_run_rpvt = pvalues_per_run_rpvt[np.logical_not(zeros_index)]
        true_pvalues = true_pvalues[np.logical_not(zeros_index)]

        # COMBI
        res_combi = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(
            pvalues_per_run_combi, true_pvalues, threshold) for threshold in tqdm(thresholds)))
        tpr_combi, _, fwer_combi, precision_combi = res_combi.T


        # T_star  - WARNING TAKES FOREVER
        tpr_permuted = 0
        fwer_permuted = 0
        precision_permuted = 0

        """
        for i in range(rep):
            with tensorflow.Session().as_default():

                model = create_montaez_dense_model_2(best_params_montaez_2)
                t_star = permuted_deepcombi_method(model, h5py_data[str(i)][:], fm_3d[str(i)][:], labels[str(i)], labels_cat[str(i)], n_permutations, alpha_sig_toy, pnorm_feature_scaling, filter_window_size, top_k, mode='all' )
                ground_truth = np.zeros((1,n_total_snps),dtype=bool)
                ground_truth[:,5000:5020] = True
                tpr, _, fwer, precision = compute_metrics(pvalues_per_run_rpvt[i], ground_truth, t_star) 
                tpr_permuted += tpr
                fwer_permuted += fwer
                precision_permuted += precision
        tpr_permuted/=rep
        fwer_permuted/=rep
        precision_permuted/=rep
        """

        # RPVT

        res_rpvt = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(
            pvalues_per_run_rpvt, true_pvalues, threshold) for threshold in tqdm(thresholds)))

        tpr_rpvt, _, fwer_rpvt, precision_rpvt = res_rpvt.T

        # Plot
        fig, axes = plt.subplots(2)
        fig.set_size_inches(18.5, 10.5)
        ax1, ax2 = axes

        ax1.set_ylim(0, 0.45)
        ax1.set_xlim(0, 0.1)

        ax1.set_ylabel('TPR')
        ax1.set_xlabel('FWER')
        ax1.plot(fwer_combi, tpr_combi, '-o',
                 label='Combi - ttbr={}'.format(ttbr))
        ax1.plot(fwer_rpvt, tpr_rpvt, '-o',
                 label='RPVT - ttbr={}'.format(ttbr))
        ax1.plot(fwer_permuted, tpr_permuted, '-x',
                 label='COMBI & permuted threshold - ttbr={}'.format(ttbr))

        ax2.set_ylabel('Precision')
        ax2.set_xlabel('TPR')
        ax2.plot(tpr_combi, precision_combi, '-o',
                 label='Combi  - ttbr={}'.format(ttbr))
        ax2.plot(tpr_rpvt, precision_rpvt, '-o',
                 label='RPVT  - ttbr={}'.format(ttbr))
        ax2.plot(tpr_permuted, precision_permuted, '-x',
                 label='COMBI & permuted threshold - ttbr={}'.format(ttbr))

        # Save results
        np.save(os.path.join(NUMPY_ARRAYS, 'combi-tpr-{}'.format(ttbr)), tpr_combi)
        np.save(os.path.join(NUMPY_ARRAYS, 'combi-fwer-{}'.format(ttbr)), fwer_combi)
        np.save(os.path.join(NUMPY_ARRAYS, 'combi-precision-{}'.format(ttbr)), precision_combi)
        np.save(os.path.join(NUMPY_ARRAYS, 'permuted-avg-tpr-pt{}'.format(ttbr)), tpr_permuted)
        np.save(os.path.join(NUMPY_ARRAYS, 'permuted-avg-fwer-pt{}'.format(ttbr)), fwer_permuted)
        np.save(os.path.join(NUMPY_ARRAYS, 'permuted-avg-precision-pt{}'.format(ttbr)), precision_permuted)

        np.save(os.path.join(NUMPY_ARRAYS, 'rpvt-tpr-{}'.format(ttbr)), tpr_rpvt)
        np.save(os.path.join(NUMPY_ARRAYS, 'rpvt-fwer-{}'.format(ttbr)), fwer_rpvt)
        np.save(os.path.join(NUMPY_ARRAYS, 'rpvt-precision-{}'.format(ttbr)), precision_rpvt)

        # CHALLENGER
        for i, window in enumerate(window_lengths):
            pvalues_challenger = pvalues_per_run_dense[:, i]

            res_dense = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(
                pvalues_challenger, true_pvalues, threshold) for threshold in tqdm(thresholds)))

            tpr_dense, _, fwer_dense, precision_dense = res_dense.T
            np.save(os.path.join(NUMPY_ARRAYS, 'tpr-{}-{}'.format(window, ttbr)), tpr_dense)
            np.save(os.path.join(NUMPY_ARRAYS, 'fwer-{}-{}'.format(window, ttbr)), fwer_dense)
            np.save(os.path.join(NUMPY_ARRAYS, 'precision-{}-{}'.format(window, ttbr)), precision_dense)
            assert fwer_combi.max() <= 1 and fwer_combi.min() >= 0
            ax1.plot(fwer_dense, tpr_dense, '-x',
                     label='Challenger  - k={}, ttbr={}'.format(window, ttbr))

            ax2.plot(tpr_dense, precision_dense, '-x',
                     label='Challenger - k={}, ttbr={}'.format(window, ttbr))

        ax1.legend()
        ax2.legend()
        fig.savefig(
            os.path.join(IMG_DIR, 'tpr_fwer_montaez2_k_coca_a1b0-bugfix-100-{}bugs.png'.format(zeros_index.sum())),
            dpi=300)

    def test_lrp_svm(self, h5py_data, fm, indices, rep, tmp_path):
        """ Compares efficiency of the combi method with several TTBR
        """
        ttbrs = [20, 6, 1, 0]
        h5py_data = h5py_data['4'][:]
        idx = indices['4']
        fig, axes = plt.subplots(len(ttbrs), 4, sharex='col')
        x_3d = fm("3d")['0'][:]
        x_2d = fm("2d")['0'][:]

        for i, ttbr in enumerate(ttbrs):
            print('Using tbrr={}'.format(ttbr))
            labels = generate_syn_phenotypes(ttbr=ttbr, quantity=rep)['4']
            l_0b = (labels + 1) / 2

            model = create_montaez_dense_model_2(best_params_montaez_2)
            model.fit(x=x_3d[idx.train],
                      y=l_0b[idx.train],
                      validation_data=(x_3d[idx.test], l_0b[idx.test]),
                      epochs=best_params_montaez_2['epochs'],
                      callbacks=[
                          ReduceLROnPlateau(monitor='val_loss',
                                            factor=best_params_montaez_2['factor'],
                                            patience=best_params_montaez_2['patience'],
                                            mode='min'),
                      ],
                      )

            model = iutils.keras.graph.model_wo_softmax(model)
            analyzer = innvestigate.analyzer.LRPAlpha1Beta0(model)
            weights = analyzer.analyze(x).sum(0)

            top_indices_sorted, postprocessed_weights = postprocess_weights(
                weights, top_k, filter_window_size, p_svm, p_pnorm_filter)

            complete_pvalues = chi_square(h5py_data, labels)
            plot_pvalues(complete_pvalues, top_indices_sorted, axes[i][0])

            # Plot distribution of relevance
            axes[i][1].plot(np.absolute(weights).reshape(-1, 3).sum(1), label='ttbr={}'.format(ttbr))
            axes[i][1].legend()
            axes[i][1].set_title('Absolute relevance')

            # Plot distribution of postprocessed vectors
            axes[i][2].plot(postprocessed_weights, label='ttbr={}'.format(ttbr))
            axes[i][2].legend()
            axes[i][2].set_title('Postprocessed relevance')

            # Plot distribution of svm weights
            svm_weights = toy_classifier.fit(x_2d, labels).coef_
            axes[i][3].plot(np.absolute(svm_weights).reshape(-1,3).sum(1), label='ttbr={}'.format(ttbr))
            axes[i][3].legend()
            axes[i][3].set_title('Absolute SVM weight')

            fig.savefig(os.path.join(IMG_DIR, 'manhattan-convdense-test.png'))
