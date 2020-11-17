import os

import matplotlib
import numpy as np
import pandas as pd
import pdb
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow
from sklearn.model_selection import ParameterGrid

from keras.callbacks import TensorBoard, ReduceLROnPlateau
from models import create_montaez_dense_model
from models import best_params_montaez
from tqdm import tqdm
from combi import combi_method
from helpers import postprocess_weights, chi_square, compute_metrics, plot_pvalues, generate_name_from_params, generate_syn_phenotypes, postprocess_weights_without_avg

from parameters_complete import random_state, nb_of_jobs, filter_window_size, p_svm, p_pnorm_filter, n_total_snps, top_k, ttbr, thresholds, IMG_DIR, NUMPY_ARRAYS
from joblib import Parallel, delayed
from combi import toy_classifier
import innvestigate
import innvestigate.utils as iutils

from sklearn.svm import LinearSVC
from sklearn.utils import class_weight
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit


from sklearn.metrics import roc_curve, precision_recall_curve


class TestDeepCOMBI(object):

    def test_indices(self, syn_labels_0based, indices):
        idx = indices['0']
        test_labels = syn_labels_0based['0'][idx.test]
        train_labels = syn_labels_0based['0'][idx.train]
        print(test_labels)
        print(train_labels)
        print(len(test_labels[test_labels == 1]))
        print(len(test_labels[test_labels == 0]))
        print(len(train_labels[train_labels == 1]))
        print(len(train_labels[train_labels == 0]))

    def test_train(self, syn_fm, syn_labels_0based, indices, rep, output_path):

        fm_ = syn_fm('3d')

        def f(x, y, idx, key):
            with tensorflow.Session().as_default():
                model = create_montaez_dense_model(best_params_montaez)
                model.fit(x=x[idx.train],
                          y=y[idx.train],
                          validation_data=(x[idx.test], y[idx.test]),
                          epochs=best_params_montaez['epochs'],
                          callbacks=[
                              ReduceLROnPlateau(monitor='val_loss',
                                                factor=best_params_montaez['factor'],
                                                patience=best_params_montaez['patience'],
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
            syn_labels_0based[str(i)],
            indices[str(i)],
            str(i)
        ) for i in range(rep))

    def test_conv_lrp(self, syn_genomic_data, syn_labels, syn_fm, syn_labels_0based, syn_labels_cat, syn_idx, rep, tmp_path):

        fig, axes = plt.subplots(1, 2 * rep, squeeze=True)
        fig.set_size_inches(10, 10)

        def f(i, x_3d, x_2d, y, y_0b, idx):
            with tensorflow.Session().as_default():
                best_params_montaez['n_snps']= x_3d[idx.train].shape[1]
                model = create_montaez_dense_model(best_params_montaez)

                y_integers = np.argmax(y_0b[idx.train], axis=1)
                class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
                d_class_weights = dict(enumerate(class_weights))
                model.fit(x=x_3d[idx.train],y=y_0b[idx.train], validation_data=(x_3d[idx.test], y_0b[idx.test]), epochs=best_params_montaez['epochs'], class_weight=d_class_weights, callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=best_params_montaez['factor'], patience=best_params_montaez['patience'],mode='min'),], verbose=1)
                model = iutils.keras.graph.model_wo_softmax(model)

                toy_classifier.fit(x_2d, y)
                svm_weights = toy_classifier.coef_[0]  # n_snps * 3
                axes[2 * i + 1].plot(np.absolute(svm_weights).reshape(-1, 3).sum(1))
                #axes[1][2 * i + 1].plot(np.absolute(svm_weights).reshape(-1, 3).sum(1))
                #axes[2][2 * i + 1].plot(np.absolute(svm_weights).reshape(-1, 3).sum(1))
                #axes[3][2 * i + 1].plot(np.absolute(svm_weights).reshape(-1, 3).sum(1))
                #axes[4][2 * i + 1].plot(np.absolute(svm_weights).reshape(-1, 3).sum(1))
                #axes[5][2 * i + 1].plot(np.absolute(svm_weights).reshape(-1, 3).sum(1))

                # LRPAlpha1Beta0
                analyzer = innvestigate.analyzer.LRPAlpha1Beta0(model)
                weights = analyzer.analyze(x_3d).sum(0)
                axes[2 * i].plot(np.absolute(weights).reshape(-1, 3).sum(1), label='a-1, b-0')

                ## LRPZ
                #analyzer = innvestigate.analyzer.LRPZ(model)
                #weights = analyzer.analyze(x_3d).sum(0)
                #axes[1][2 * i].plot(np.absolute(weights).reshape(-1, 3).sum(1), #label='ttbr={},lrpz'.format(ttbr))

                ## LRPEpsilon
                #analyzer = innvestigate.analyzer.LRPEpsilon(model, epsilon=1e-5)
                #weights = analyzer.analyze(x_3d).sum(0)
                #axes[2][2 * i].plot(np.absolute(weights).reshape(-1, 3).sum(1), label='epsilon')

                ## LRPAlpha2Beta1
                #analyzer = innvestigate.analyzer.LRPAlpha2Beta1(model)
                #weights = analyzer.analyze(x_3d).sum(0)
                #axes[3][2 * i].plot(np.absolute(weights).reshape(-1, 3).sum(1), label='a-2, b-1')

                ## LRPZ+
                #analyzer = innvestigate.analyzer.LRPZPlus(model)
                #weights = analyzer.analyze(x_3d).sum(0)
                #axes[4][2 * i].plot(np.absolute(weights).reshape(-1, 3).sum(1), label='lrpzplus')

                ## LRPAlpha1Beta0IgnoreBias
                #analyzer = innvestigate.analyzer.LRPAlpha1Beta0IgnoreBias(model)
                #weights = analyzer.analyze(x_3d).sum(0)
                #axes[5][2 * i].plot(np.absolute(weights).reshape(-1, 3).sum(1), label='LRPAlpha1Beta0IgnoreBias')

        Parallel(n_jobs=-1, prefer="threads")(delayed(f)(i, syn_fm("3d")[str(i)][:], syn_fm("2d")[str(i)][:], syn_labels[str(i)], syn_labels_cat[str(i)], syn_idx[str(i)])for i in range(rep))

        fig.savefig(os.path.join(IMG_DIR, 'montaez-lrp-vs-svmweights.png'))

    def test_hp_params(self, syn_fm, syn_labels_0based, syn_labels_cat, indices, rep, output_path):
        syn_fm = syn_fm('3d')

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
                model = create_montaez_dense_model(g)

                histories = [model.fit(x=fm[indices[str(i)].train],
                                       y=syn_labels_cat[str(i)][indices[str(i)].train],
                                       validation_data=(
                                           fm[indices[str(i)].test], syn_labels_cat[str(i)][indices[str(i)].test]),
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

    #def test_svm_cnn_comparison(self, syn_fm, syn_labels, syn_labels_cat, rep, indices):
    def test_svm_cnn_comparison(self, syn_fm, syn_labels, syn_labels_cat, rep, syn_idx):
        """ Compares performance of SVM and CNN models
        """
        fm_3d = syn_fm("3d")
		
        """fm[fm>0]=1
        fm[fm<0]=0.

        mean = np.mean(np.mean(fm,0),0)
        std = np.std(np.std(fm,0),0)

        fm = (fm-mean)/std
        """
		
        fm_2d = syn_fm("2d")
		
        def fit_cnn(x, y, idx):
            #'batch_size': 32,'factor':0.7125,'patience':50,
            best_params_montaez = {'epochs': 500,  'l1_reg': 0.001, 'l2_reg': 0.0001,'lr' :1e-05, 'dropout_rate':0.3,   'hidden_neurons':64, 'n_snps': x.shape[1]}
            model = create_montaez_dense_model(best_params_montaez)
            y_integers = np.argmax(y[idx.train], axis=1)
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
            d_class_weights = dict(enumerate(class_weights))

            return model.fit(x=x[idx.train], y=y[idx.train], validation_data=(x[idx.test], y[idx.test]), epochs=best_params_montaez['epochs'], callbacks=[ ReduceLROnPlateau(monitor='val_loss', mode='min')],class_weight=d_class_weights).history['val_categorical_accuracy'][-1]

        def fit_svm(x, y, idx):
            toy_classifier = LinearSVC(penalty='l2', loss='hinge', C=1.0000e-05, dual=True, tol=1e-3, verbose=0)

            svm_model = toy_classifier.fit(x[idx.train], y[idx.train])
            return svm_model.score(x[idx.test], y[idx.test])

        svm_val_acc = Parallel(n_jobs=30)(delayed(fit_svm)(fm_2d[str(i)][:], syn_labels[str(i)], syn_idx[str(i)]) for i in tqdm(range(1, rep)))
        cnn_val_acc = Parallel(n_jobs=30)(delayed(fit_cnn)(fm_3d[str(i)][:], syn_labels_cat[str(i)], syn_idx[str(i)]) for i in tqdm(range(1, rep)))

        cnn_val_acc = cnn_val_acc * 100
        svm_val_acc = svm_val_acc * 100

        # 3. Compare average val_acc on all syn datasets
        print('SVM val_acc mean {} / std {} / max {} / min {}'.format(np.mean(svm_val_acc), np.std(svm_val_acc), np.max(svm_val_acc), np.min(svm_val_acc)))
        print('cnn val_acc mean {} / std {} / max {} / min {}'.format(np.mean(cnn_val_acc), np.std(cnn_val_acc), np.max(cnn_val_acc), np.min(cnn_val_acc)))
        pdb.set_trace()


    def test_svm_cnn_comparison_alex(self, syn_fm, syn_labels, syn_labels_cat, rep, syn_idx):
        """ Compares performance of SVM and CNN models
        """

        fm_3d = syn_fm("3d")
        fm_2d = syn_fm("2d")

        def fit_cnn(x, y, idx):
            best_params_montaez['n_snps']= x.shape[1]
            model = create_montaez_dense_model(best_params_montaez)
            y_integers = np.argmax(y[idx.train], axis=1)
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
            d_class_weights = dict(enumerate(class_weights))
            return model.fit(x=x[idx.train], y=y[idx.train], validation_data=(x[idx.test], y[idx.test]), epochs=best_params_montaez['epochs'],class_weight=d_class_weights, callbacks=[ReduceLROnPlateau(monitor='val_loss',  factor=best_params_montaez['factor'], patience=best_params_montaez['patience'], mode='min')]).history['val_weighted_categorical_accuracy'][-1]
			# change if you want accuracy or balanced accuracy: val_categorical_accuracy, val_weighted_categorical_accuracy, AUC(not working... would need higher keras version)

        def fit_svm(x, y, idx):
            # if you want to optimize c:
            #c_candidates = np.logspace(-7, -1, num=20)
            #scores=np.zeros(len(c_candidates))
            #for i in np.arange(len(c_candidates)):
            #    c = c_candidates[i]
            #    clf = LinearSVC(C=c, penalty='l2', loss='hinge', tol= 1e-3,dual=True, verbose=0, class_weight='balanced')
            #    scores[i] = np.mean(cross_val_score(clf, x, y, cv=StratifiedShuffleSplit(n_splits=5), scoring='accuracy'))
            #print(c_candidates, scores)			
            #c_max = c_candidates[np.argmax(scores)]
            c_max = 0.0022
            toy_classifier = LinearSVC(C=c_max, penalty='l2', loss='hinge', tol= 1e-3,dual=True, verbose=0, class_weight='balanced')
            bla = cross_val_score(toy_classifier, x, y, cv=StratifiedShuffleSplit(n_splits=5), scoring='balanced_accuracy')
            #change if you want accuracy or balanced accuracy: scoring = accuracy, balanced_accuracy, roc_auc, average_precision_score
            return np.mean(bla)

        svm_val_acc = Parallel(n_jobs=1)(delayed(fit_svm)(fm_2d[str(i)][:], syn_labels[str(i)], syn_idx[str(i)]) for i in tqdm(range(1, rep)))
        cnn_val_acc = Parallel(n_jobs=1)(delayed(fit_cnn)(fm_3d[str(i)][:], syn_labels_cat[str(i)], syn_idx[str(i)]) for i in tqdm(range(1, rep)))

        cnn_val_acc = cnn_val_acc * 100
        svm_val_acc = svm_val_acc * 100

        # 3. Compare average val_acc on all syn datasets
        print('SVM val_acc mean {} / std {} / max {} / min {}'.format(np.mean(svm_val_acc), np.std(svm_val_acc), np.max(svm_val_acc), np.min(svm_val_acc)))
        print('cnn val_acc mean {} / std {} / max {} / min {}'.format(np.mean(cnn_val_acc), np.std(cnn_val_acc), np.max(cnn_val_acc), np.min(cnn_val_acc)))

    def test_tpr_fwer(self, syn_genomic_data, syn_labels, syn_labels_0based, syn_labels_cat, syn_fm, syn_idx, rep, syn_true_pvalues):
        """ Compares combi vs dense curves
        """

        window_lengths = [35]

        best_params_montaez = {'epochs': 500,  'l1_reg': 0.001, 'l2_reg': 0.0001,'lr' :1e-05, 'dropout_rate':0.3,   'hidden_neurons':64, 'n_snps': n_total_snps}

        # n_permutations = 2

        def combi_compute_pvalues(d, x, fm, l,filter_window_size,pf,ps,k):
            #clf, syn_genomic_data[str(i)][:], fm_2d[str(i)][:], syn_labels[str(i)], 35, 2, 2, 30
            idx, pvalues, _ = combi_method(d, x,fm, l,filter_window_size,pf,ps,k)
			#combi_method(classifier,data, fm, labels, filter_window_size, pnorm_filter, psvm, top_k)
            pvalues_filled = np.ones(n_total_snps)
            pvalues_filled[idx] = pvalues
            del d, l
            return pvalues_filled

        def challenger_compute_pvalues(d, x, l_0b, l, idx):
            is_only_zeros = False
            with tensorflow.Session().as_default():

                model = create_montaez_dense_model(best_params_montaez)

                model.fit(x=x[idx.train], y=l_0b[idx.train],
                          validation_data=(x[idx.test], l_0b[idx.test]),
                          epochs=best_params_montaez['epochs'],
                          callbacks=[
                              ReduceLROnPlateau(monitor='val_loss',
                                                mode='min'),
                          ])

                model = iutils.keras.graph.model_wo_softmax(model)
                analyzer = innvestigate.analyzer.LRPAlpha2Beta1(model)
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

        fm_2d = syn_fm("2d")
        fm_3d = syn_fm("3d")
        clf = LinearSVC(penalty='l2', loss='hinge', C=1.0000e-05, dual=True, tol=1e-3, verbose=0)

        pvalues_per_run_combi = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(
            combi_compute_pvalues)(clf, syn_genomic_data[str(i)][:], fm_2d[str(i)][:], syn_labels[str(i)], 35, 2, 2, 30) for i in tqdm(range(rep))))

        pvalues_per_run_rpvt = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(
            chi_square)(syn_genomic_data[str(i)][:], syn_labels[str(i)]) for i in tqdm(range(rep))))

        # len(thresholds) * len(window_sizes) * 10020
        a = Parallel(n_jobs=-1, require='sharedmem')(delayed(
            challenger_compute_pvalues)(syn_genomic_data[str(i)][:], fm_3d[str(i)][:], syn_labels_cat[str(i)], syn_labels[str(i)], syn_idx[str(i)]) for i in tqdm(range(rep)))

        # INNvestigate bugfix
        zeros_index = np.array(list(np.array(a)[:, 1]))
        pvalues_per_run_dense = np.array(list(np.array(a)[:, 0]))

        pvalues_per_run_combi = pvalues_per_run_combi[np.logical_not(zeros_index)]
        pvalues_per_run_dense = pvalues_per_run_dense[np.logical_not(zeros_index)]
        pvalues_per_run_rpvt = pvalues_per_run_rpvt[np.logical_not(zeros_index)]
        true_pvalues = syn_true_pvalues[np.logical_not(zeros_index)]

        # COMBI
        res_combi = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(pvalues_per_run_combi, true_pvalues, threshold) for threshold in tqdm(thresholds)))
        tpr_combi, _, fwer_combi, precision_combi = res_combi.T


        # T_star  - WARNING TAKES FOREVER
        tpr_permuted = 0
        fwer_permuted = 0
        precision_permuted = 0

        """
        for i in range(rep):
            with tensorflow.Session().as_default():

                model = create_montaez_dense_model_2(best_params_montaez_2)
                t_star = permuted_deepcombi_method(model, h5py_data[str(i)][:], fm_3d[str(i)][:], labels[str(i)], labels_cat[str(i)], n_permutations, alpha_sig_toy, filter_window_size, top_k, mode='all' )
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
                 label='Combi')
        ax1.plot(fwer_rpvt, tpr_rpvt, '-o',
                 label='RPVT')
        #ax1.plot(fwer_permuted, tpr_permuted, '-x',
        #         label='COMBI & permuted threshold - ttbr={}'.format(ttbr))

        ax2.set_ylabel('Precision')
        ax2.set_xlabel('TPR')
        ax2.plot(tpr_combi, precision_combi, '-o',
                 label='Combi')
        ax2.plot(tpr_rpvt, precision_rpvt, '-o',
                 label='RPVT')
        #ax2.plot(tpr_permuted, precision_permuted, '-x',
        #         label='COMBI & permuted threshold - ttbr={}'.format(ttbr))

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
            ax1.plot(fwer_dense, tpr_dense, '-x', label='DeepCOMBI')
            ax2.plot(tpr_dense, precision_dense, '-x', label='DeepCOMBI')

        ax1.legend()
        ax2.legend()
        fig.savefig(
            os.path.join(IMG_DIR, 'tpr_fwer_montaez_combi_newsettings.png'.format(zeros_index.sum())),
            dpi=300)




    def test_tpr_fwer_alex(self, syn_genomic_data, syn_labels, syn_labels_0based, syn_labels_cat, syn_fm, syn_idx, rep, syn_true_pvalues):
        """ Compares combi vs dense curves
        """

        window_length = 35
        best_params_montaez['n_snps'] = n_total_snps

        def combi_compute_pvalues(d, x, fm, l,filter_window_size,pf,ps,k):

            idx, pvalues, raw_weights = combi_method(d, x,fm, l,filter_window_size,pf,ps,k)
            pvalues_filled = np.ones(n_total_snps)
            pvalues_filled[idx] = pvalues
            raw_weights = postprocess_weights_without_avg(raw_weights, p_svm)
			
            # Map the raw weights to look like p-values between 0 and 1 (reverse order)		
            # Figure out how 'wide' range is
            leftSpan = np.max(raw_weights) - np.min(raw_weights)

            # Convert the left range into a 0-1 range (float)
            valueScaled = (raw_weights - np.min(raw_weights)) / leftSpan

            # Reverse order
            raw_weights = 1 - valueScaled	
			
            del d, l
            return pvalues_filled, raw_weights

        def challenger_compute_pvalues(d, x, l_0b, l, idx):
            is_only_zeros = False
            with tensorflow.Session().as_default():

                model = create_montaez_dense_model(best_params_montaez)
                y_integers = np.argmax(l_0b[idx.train], axis=1)
                class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
                d_class_weights = dict(enumerate(class_weights))
                model.fit(x=x[idx.train], y=l_0b[idx.train], validation_data=(x[idx.test], l_0b[idx.test]), epochs=best_params_montaez['epochs'],class_weight=d_class_weights, callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=best_params_montaez['factor'], patience=best_params_montaez['patience'], mode='min'),])

                model = iutils.keras.graph.model_wo_softmax(model)
                analyzer = innvestigate.analyzer.LRPAlpha1Beta0(model)
                weights = analyzer.analyze(x).sum(0)

                if np.max(abs(weights)) < 0.005:
                    fig, axes = plt.subplots(1)
                    is_only_zeros = True
                    axes.plot(np.absolute(weights).sum(axis=1))
                    fig.savefig(os.path.join(IMG_DIR, 'test.png'))

                top_indices_sorted, _ = postprocess_weights(weights, top_k, window_length, p_svm, p_pnorm_filter)
                rawlrp_scores_now = postprocess_weights_without_avg(weights,p_svm)
				
                # Map the raw weights to look like p-values between 0 and 1 (reverse order)		
                # Figure out how 'wide' range is
                leftSpan = np.max(rawlrp_scores_now) - np.min(rawlrp_scores_now)

                # Convert the left range into a 0-1 range (float)
                valueScaled = (rawlrp_scores_now - np.min(rawlrp_scores_now)) / leftSpan

                # Reverse order
                rawlrp_scores_now = 1 - valueScaled					
				
                pvalues = chi_square(d[:, top_indices_sorted], l)
                pvalues_filled = np.ones(n_total_snps)
                pvalues_filled[top_indices_sorted] = pvalues
                del d, x, l

            return pvalues_filled, is_only_zeros, rawlrp_scores_now

        fm_2d = syn_fm("2d")
        fm_3d = syn_fm("3d")

        clf = LinearSVC(penalty='l2', loss='hinge', C=0.0022, dual=True, tol=1e-3, verbose=0, class_weight='balanced')

        bla = Parallel(n_jobs=-1, require='sharedmem')(delayed(combi_compute_pvalues)(clf, syn_genomic_data[str(i)][:], fm_2d[str(i)][:], syn_labels[str(i)], 35, 2, 2, 30) for i in tqdm(range(rep)))
        raw_svmweights_per_run_combi = np.array(list(np.array(bla)[:, 1]))
        pvalues_per_run_combi = np.array(list(np.array(bla)[:, 0]))

        pvalues_per_run_rpvt = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(chi_square)(syn_genomic_data[str(i)][:], syn_labels[str(i)]) for i in tqdm(range(rep))))

        # len(thresholds) * len(window_sizes) * 10020
        abl  = Parallel(n_jobs=-1, require='sharedmem')(delayed(challenger_compute_pvalues)(syn_genomic_data[str(i)][:], fm_3d[str(i)][:], syn_labels_cat[str(i)], syn_labels[str(i)], syn_idx[str(i)]) for i in tqdm(range(rep)))

        # Collect results
        pvalues_per_run_dense = np.array(list(np.array(abl)[:, 0]))
        rawlrp_scores_per_run_dense = np.array(list(np.array(abl)[:, 2]))
  
        # INNvestigate bugfix
        zeros_index = np.array(list(np.array(abl)[:, 1]))  
        pvalues_per_run_combi = pvalues_per_run_combi[np.logical_not(zeros_index)]
        raw_svmweights_per_run_combi = raw_svmweights_per_run_combi[np.logical_not(zeros_index)]
        pvalues_per_run_dense = pvalues_per_run_dense[np.logical_not(zeros_index)]
        rawlrp_scores_per_run_dense = rawlrp_scores_per_run_dense[np.logical_not(zeros_index)]
        pvalues_per_run_rpvt = pvalues_per_run_rpvt[np.logical_not(zeros_index)]
        true_pvalues = syn_true_pvalues[np.logical_not(zeros_index)]

        # COMBI
        res_combi = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(pvalues_per_run_combi, true_pvalues, threshold) for threshold in tqdm(thresholds)))
        tpr_combi, _, fwer_combi, precision_combi = res_combi.T

        # SVM weights
        res_rawsvm = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(raw_svmweights_per_run_combi, true_pvalues, threshold) for threshold in tqdm(thresholds)))
        tpr_rawsvm, _, fwer_rawsvm, precision_rawsvm = res_rawsvm.T

        # RPVT
        res_rpvt = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(pvalues_per_run_rpvt, true_pvalues, threshold) for threshold in tqdm(thresholds)))
		
        tpr_rpvt, _, fwer_rpvt, precision_rpvt = res_rpvt.T

        # Plot
        fig, axes = plt.subplots(1,2)

        fig.set_size_inches(15, 9)
        ax1, ax2 = axes

        ax1.set_ylim(0, 0.7)
        ax1.set_xlim(0, 0.3)
        ax1.set_ylabel('True positive rate', fontsize=14)
        ax1.set_xlabel('Family-wise error rate', fontsize=14)
        ax2.set_ylabel('Precision', fontsize=14)
        ax2.set_xlabel('True positive rate', fontsize=14)

        # CURVES must stop somewhere
        tpr_rpvt_new = tpr_rpvt[tpr_rpvt < 1]
        fwer_rpvt = fwer_rpvt[tpr_rpvt < 1]
        precision_rpvt = precision_rpvt[tpr_rpvt < 1]
        tpr_rpvt = tpr_rpvt_new

        tpr_combi_new = tpr_combi[tpr_combi < 1]
        fwer_combi = fwer_combi[tpr_combi < 1]
        precision_combi = precision_combi[tpr_combi < 1]	
        tpr_combi = tpr_combi_new

        tpr_rawsvm_new = tpr_rawsvm[tpr_rawsvm < 1]
        fwer_rawsvm = fwer_rawsvm[tpr_rawsvm < 1]
        precision_rawsvm = precision_rawsvm[tpr_rawsvm < 1]
        tpr_rawsvm = tpr_rawsvm_new
		
        # RPVT
        ax1.plot(fwer_rpvt, tpr_rpvt, label='RPVT', color='lightsteelblue', linewidth=2)
        ax2.plot(tpr_rpvt, precision_rpvt, color='lightsteelblue', label='RPVT', linewidth=2)

        # COMBI 
        ax1.plot(fwer_combi, tpr_combi, color='darkblue', label='COMBI', linewidth=2)
        ax2.plot(tpr_combi, precision_combi, color='darkblue', label='COMBI', linewidth=2)
 
        # raw SVM weights
        ax1.plot(fwer_rawsvm, tpr_rawsvm, linestyle='--', color='darkblue', label='SVM weights', linewidth=2)
        ax2.plot(tpr_rawsvm, precision_rawsvm, linestyle='--', color='darkblue', label='SVM weights', linewidth=2)

        # DeepCOMBI + LRP scores
        res_dense = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(pvalues_per_run_dense, true_pvalues, threshold) for threshold in tqdm(thresholds)))
        res_rawlrp_dense = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(rawlrp_scores_per_run_dense, true_pvalues, threshold) for threshold in tqdm(thresholds)))

        tpr_dense, _, fwer_dense, precision_dense = res_dense.T
        tpr_rawlrp_dense, _, fwer_rawlrp_dense, precision_rawlrp_dense = res_rawlrp_dense.T

        assert fwer_combi.max() <= 1 and fwer_combi.min() >= 0
        tpr_dense_new = tpr_dense[tpr_dense < 1]
        fwer_dense = fwer_dense[tpr_dense < 1]
        precision_dense = precision_dense[tpr_dense < 1]
        tpr_dense = tpr_dense_new

        tpr_rawlrp_dense_new = tpr_rawlrp_dense[tpr_rawlrp_dense < 1]
        fwer_rawlrp_dense = fwer_rawlrp_dense[tpr_rawlrp_dense < 1]
        precision_rawlrp_dense = precision_rawlrp_dense[tpr_rawlrp_dense < 1]
        tpr_rawlrp_dense = tpr_rawlrp_dense_new

        # DeepCOMBI
        ax1.plot(fwer_dense, tpr_dense, color='fuchsia', label='DeepCOMBI', linewidth=3)
        ax2.plot(tpr_dense, precision_dense, color='fuchsia', label='DeepCOMBI', linewidth=3)

        # LRP scores
        ax1.plot(fwer_rawlrp_dense, tpr_rawlrp_dense, color='fuchsia', linestyle='--', label='LRP scores', linewidth=2)
        ax2.plot(tpr_rawlrp_dense, precision_rawlrp_dense, color='fuchsia', linestyle='--', label='LRP scores', linewidth=2)

        ax1.legend(fontsize=14,loc= 'lower right')
        ax2.legend(fontsize=14, loc= 'lower right')
        fig.savefig(os.path.join(IMG_DIR, 'tpr_fwer_montaez_final1000_NAR.png'), bbox_inches='tight', dpi=300)
        print(np.sum(zeros_index))
        pdb.set_trace()

        # CURVES must stop somewhere
        #combi_fp = combi_fp[combi_fp < 80]
        #combi_tp = combi_tp[:len(combi_fp)]
        #deepcombi_fp = deepcombi_fp[deepcombi_fp < 80]
        #deepcombi_tp = deepcombi_tp[:len(deepcombi_fp)]



    def test_lrp_svm(self, syn_genomic_data, syn_fm, syn_idx, rep, tmp_path, syn_true_pvalues):
        """ Compares efficiency of the combi method with several TTBR
        """ 
        rep_to_plot = 0
        ttbrs = [0.5, 1,1.5]
        idx = syn_idx[str(rep_to_plot)]
        fig, axes = plt.subplots(len(ttbrs), 5, figsize=[30,15])
        x_3d = syn_fm("3d")[str(rep_to_plot)][:]
        x_2d = syn_fm("2d")[str(rep_to_plot)][:]
        indices_true= [inds_true for inds_true, x in enumerate(syn_true_pvalues[0].flatten()) if x]

        for i, ttbr in enumerate(ttbrs):
            print('Using tbrr={}'.format(ttbr))
            labels = generate_syn_phenotypes(tower_to_base_ratio=ttbr, quantity=rep)
            labels_cat = {}
            for key, l in labels.items():
                labels_cat[key] = tensorflow.keras.utils.to_categorical((l+1)/2)
            
            best_params_montaez['n_snps']= x_3d.shape[1]
            
            l_0b=labels_cat[str(rep_to_plot)]

            model = create_montaez_dense_model(best_params_montaez)
            y_integers = np.argmax(l_0b[idx.train], axis=1)
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
            d_class_weights = dict(enumerate(class_weights))

            model.fit(x=x_3d[idx.train], y=l_0b[idx.train], validation_data=(x_3d[idx.test], l_0b[idx.test]), epochs=best_params_montaez['epochs'], class_weight=d_class_weights, callbacks=[ ReduceLROnPlateau(monitor='val_loss', factor=best_params_montaez['factor'], patience=best_params_montaez['patience'], mode='min'),],)

            model = iutils.keras.graph.model_wo_softmax(model)
            analyzer = innvestigate.analyzer.LRPAlpha1Beta0(model)
            weights = analyzer.analyze(x_3d).sum(0)

            top_indices_sorted, filtered_weights  = postprocess_weights(weights, top_k, filter_window_size, p_svm, p_pnorm_filter)

            complete_pvalues = chi_square(syn_genomic_data[str(rep_to_plot)][:], labels[str(rep_to_plot)])
            
            pvalues_filled_deep = np.ones(n_total_snps)
            pvalues_filled_deep[top_indices_sorted] = complete_pvalues[top_indices_sorted]

            # Plot RPVT
            plot_pvalues(complete_pvalues, indices_true, axes[i][0])
            if i==0:
                axes[i][0].set_title('RPVT $-log_{10}$(p-values)', fontsize=22)
            axes[i][0].set_ylabel('$-log_{10}$(p-value)', fontsize=18)
            plt.setp(axes[i][0].get_yticklabels(), fontsize=16)
            plt.setp(axes[i][0].get_xticklabels(), fontsize=16)

            # Plot svm weights 
            clf = LinearSVC(penalty='l2', loss='hinge', C=0.0022, dual=True, tol=1e-3, verbose=0, class_weight='balanced')
            idx_now, pvalues, raw_weights = combi_method(clf, syn_genomic_data[str(rep_to_plot)][:],x_2d, labels[str(rep_to_plot)],  35, 2, 2, 30)
            #filtered_svm_weights = postprocess_weights_without_avg(raw_weights, p_svm)
            pvalues_filled_combi = np.ones(len(complete_pvalues))
            pvalues_filled_combi[idx_now] = pvalues
            #svm_weights = toy_classifier.fit(x_2d, labels[str(rep_to_plot)]).coef_
            axes[i][1].scatter(range(len(np.absolute(raw_weights).sum(1))), 1000*np.absolute(raw_weights).sum(1), marker='.', color='darkblue')
            axes[i][1].scatter(indices_true,1000*np.absolute(raw_weights).sum(1)[indices_true], color='fuchsia')
            axes[i][1].set_ylim(0,1000*(np.max(np.absolute(raw_weights).sum(1))+0.001))
            if i==0:
                axes[i][1].set_title('Absolute SVM weights * 1000', fontsize=22)
            plt.setp(axes[i][1].get_yticklabels(), fontsize=16)
            plt.setp(axes[i][1].get_xticklabels(), fontsize=16)
			
            # Plot COMBI
            plot_pvalues(pvalues_filled_combi, indices_true, axes[i][2])
            if i==0:
                axes[i][2].set_title('COMBI $-log_{10}$(p-values)', fontsize=22)
            if i==2:
                axes[i][2].set_xlabel('SNP position', fontsize=18)
            plt.setp(axes[i][2].get_yticklabels(), fontsize=16)
            plt.setp(axes[i][2].get_xticklabels(), fontsize=16)
			
            # Plot LRP relevance scores
            axes[i][3].scatter(range(len(np.absolute(weights).reshape(-1, 3).sum(1))), np.absolute(weights).reshape(-1, 3).sum(1), marker='.', color='darkblue')
            axes[i][3].scatter(indices_true,np.absolute(weights).reshape(-1, 3).sum(1)[indices_true], color='fuchsia')
            #axes[i][1].legend()
            axes[i][3].set_ylim(0,np.max(np.absolute(weights).reshape(-1, 3).sum(1))+1)
            if i==0:
                axes[i][3].set_title('LRP relevance scores', fontsize=22)
            plt.setp(axes[i][3].get_yticklabels(), fontsize=16)
            plt.setp(axes[i][3].get_xticklabels(), fontsize=16)
			
            # Plot DeepCOMBI
            plot_pvalues(pvalues_filled_deep, indices_true, axes[i][4])
            if i==0:
                axes[i][4].set_title('DeepCOMBI $-log_{10}$(p-value)', fontsize=22)
            plt.setp(axes[i][4].get_yticklabels(), fontsize=16)
            plt.setp(axes[i][4].get_xticklabels(), fontsize=16)
			
            ## Plot distribution of postprocessed vectors
            #axes[i][2].plot(postprocessed_weights)
            #axes[i][2].set_title('Postprocessed relevance')

            fig.savefig(os.path.join(IMG_DIR, 'manhattan-example-toy-NAR.png'), bbox_inches='tight')

    def test_svm_cnn_comparison_alexnew(self, syn_fm, syn_labels,syn_labels_0based, syn_labels_cat, rep, syn_idx):
        fm_2d = syn_fm("2d")

        def fit_svm(x, y, idx):
            Cs= 0.0022  
            svm_epsilon = 1e-3
            toy_classifier = LinearSVC(C=Cs, penalty='l2', loss='hinge', tol= svm_epsilon,dual=True, verbose=0)
            svm_model = toy_classifier.fit(x[idx.train], y[idx.train])
            return svm_model.score(x[idx.test], y[idx.test])

        svm_val_acc = Parallel(
            n_jobs=30)(delayed(fit_svm)(fm_2d[str(i)][:], syn_labels[str(i)], syn_idx[str(i)]) for i in tqdm(range(1, rep)))
        svm_val_acc = svm_val_acc * 100

        # 3. Compare average val_acc on all syn datasets
        print('SVM val_acc mean {}/std{}/max{}/min{}'.format(
            np.mean(svm_val_acc), np.std(svm_val_acc), np.max(svm_val_acc), np.min(svm_val_acc)))
        print(svm_val_acc[0:rep])
        print(len(svm_val_acc))

    def test_svm_cnn_comparison_mysvm(self, syn_fm, syn_labels, syn_labels_cat, rep, syn_idx):
        fm_2d = syn_fm("2d")
        def fit_svm(x, y, idx):
            toy_classifier = LinearSVC(C=0.0022, penalty='l2', loss='hinge', tol= 1e-3,dual=True, verbose=0)
            #svm_model = toy_classifier.fit(x[idx.train], y[idx.train])
            #return svm_model.score(x[idx.test], y[idx.test])
            bla = cross_val_score(toy_classifier, x, y, cv=5, scoring='balanced_accuracy')
            #scoring = accuracy, balanced_accuracy, roc_auc, average_precision_score
            return bla[-1]

        svm_val_acc = Parallel(n_jobs=1)(delayed(fit_svm)(fm_2d[str(i)][:], syn_labels[str(i)], syn_idx[str(i)]) for i in tqdm(range(1, rep)))

        svm_val_acc = svm_val_acc * 100

        # 3. Compare average val_acc on all syn datasets
        print('SVM val_acc mean {} / std {} / max {} / min {}'.format(np.mean(svm_val_acc), np.std(svm_val_acc), np.max(svm_val_acc), np.min(svm_val_acc)))
        print(svm_val_acc[0:rep])
        print(len(svm_val_acc))
