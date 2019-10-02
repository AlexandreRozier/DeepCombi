import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import pickle
import numpy as np
import tensorflow
from models import create_montaez_dense_model_2, best_params_montaez_2
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from helpers import char_matrix_to_featmat, postprocess_weights, get_available_gpus
from parameters_complete import FINAL_RESULTS_DIR, IMG_DIR, real_pnorm_feature_scaling, real_p_pnorm_filter, filter_window_size, real_top_k, p_svm
from parameters_complete import disease_IDs

from keras.models import load_model
from combi import permuted_deepcombi_method, combi_method, real_classifier
from joblib import Parallel, delayed
from tqdm import tqdm
from helpers import chi_square
import talos
from talos.utils.gpu_utils import parallel_gpu_jobs

import innvestigate
import innvestigate.utils as iutils

TEST_PERCENTAGE = 0.20


class TestLOTR(object):

    def test_lrp(self, real_h5py_data, real_labels, real_labels_0based, real_labels_cat, real_idx, rep, tmp_path):

        fig, axes = plt.subplots(5, 1, sharex='col')

        h5py_data = real_h5py_data(1)[:]

        x_2d = char_matrix_to_featmat(h5py_data, '2d')
        x_3d = char_matrix_to_featmat(h5py_data, '3d')

        g = best_params_montaez_2
        g['n_snps'] = x_3d.shape[1]


        model = create_montaez_dense_model_2(g)
        model.fit(x=x_3d[real_idx.train],
                  y=real_labels_cat[real_idx.train],
                  validation_data=(x_3d[real_idx.test], real_labels_cat[real_idx.test]),
                  epochs=g['epochs']
                  )

        model = iutils.keras.graph.model_wo_softmax(model)
        analyzer = innvestigate.analyzer.LRPAlpha1Beta0(model)
        weights = np.absolute(analyzer.analyze(x_3d).sum(0))

        _, postprocessed_weights = postprocess_weights(
            weights, real_top_k, filter_window_size, p_svm, real_p_pnorm_filter)

        complete_pvalues = chi_square(h5py_data, real_labels)
        axes[0].plot(-np.log10(complete_pvalues))
        axes[0].set_title('RPV')

        # Plot distribution of relevance
        axes[1].plot(np.absolute(weights).reshape(-1, 3).sum(1))
        axes[1].set_title('Absolute relevance')

        # Plot distribution of postprocessed vectors
        axes[2].plot(postprocessed_weights)
        axes[2].set_title('Postprocessed relevance')

        # Plot distribution of svm weights
        svm_weights = real_classifier.fit(x_2d, real_labels).coef_
        axes[3].plot(np.absolute(svm_weights).reshape(-1, 3).sum(1))
        axes[3].set_title('Absolute SVM weight')

        _, postprocessed_svm_weights = postprocess_weights(
            np.absolute(svm_weights), real_top_k, filter_window_size, p_svm, real_p_pnorm_filter)

        axes[4].plot(postprocessed_svm_weights)
        axes[4].set_title('Postprocessed SVM weight')

        fig.savefig(os.path.join(IMG_DIR, 'manhattan-real.png'))

    def test_svm_avg_accuracy(self, real_h5py_data, real_labels, real_idx):
        score = 0
        chroms = [1, 2, 3, 5]
        for chrom in tqdm(chroms):
            data = real_h5py_data(chrom)[:]
            x_2d = char_matrix_to_featmat(data, '2d')

            real_classifier.fit(x_2d[real_idx.train], real_labels[real_idx.train])
            score += real_classifier.score(x_2d[real_idx.test], real_labels[real_idx.test])
        score /= len(chroms)
        print("Average SVM test accuracy : {}".format(score))

    def test_class_proportions(self, real_labels, real_idx):
        train_labels = real_labels[real_idx.train]
        test_labels = real_labels[real_idx.test]

        print("Train: Cases:{};  Controls:{}".format((train_labels > 0.5).sum(), (train_labels < 0.5).sum()))
        print("Test: Cases:{};  Controls:{}".format((test_labels > 0.5).sum(), (test_labels < 0.5).sum()))

    def test_train_networks(self, real_h5py_data, real_labels, real_labels_0based, real_labels_cat, real_idx):
        """ Runs HP search for a subset of chromosomes (on CPU, high degree of paralellism.)
        """
        # Each node gets a set of chromosomes to process :D

        chrom = int(os.environ['SGE_TASK_ID'])
        print('loading..')

        # 1. Do hyperparam search on each chromosome and find parameters with BEST VAL ACCURAC

        data = real_h5py_data(chrom)[:]
        fm = char_matrix_to_featmat(data, '3d')

        print('loaded')

        params_space = {
            'n_snps': [fm.shape[1]],
            'epochs': [800],
            'dropout_rate': [0.3],
            'l1_reg': list(np.logspace(-7, -2, 6)),
            'l2_reg': [0],
            'hidden_neurons': [3, 6, 10],
            'lr': list(np.logspace(-4, -1, 4)),
        }

        def talos_wrapper(x, y, x_val, y_val, params):
            model = create_montaez_dense_model_2(params)
            out = model.fit(x=x,
                            y=y,
                            validation_data=(x_val, y_val),
                            epochs=params['epochs']
                            )
            return out, model

        nb_gpus = get_available_gpus()

        if nb_gpus == 1:
            parallel_gpu_jobs(0.33)

        talos.Scan(x=fm[real_idx.train],
                   y=real_labels_cat[real_idx.train],
                   x_val=fm[real_idx.test],
                   y_val=real_labels_cat[real_idx.test],
                   # reduction_method='correlation',
                   # reduction_interval=10,
                   # reduction_window=10,
                   # reduction_metric='val_acc',
                   # reduction_threshold=0.2,
                   minimize_loss=False,
                   params=params_space,
                   model=talos_wrapper,
                   experiment_name='talos_chrom_{}'.format(chrom))

    def test_permutations(self, real_h5py_data, real_labels, real_labels_0based, real_labels_cat, real_idx, alphas,
                          alphas_EV):
        """ Computes t_star for each chromosome thanks to the permutation method.
        """

        chrom = int(os.environ['SGE_TASK_ID'])

        # 1. Do hyperparam search on each chromosome and find parameters with BEST VAL ACCURAC

        data = real_h5py_data(chrom)[:]

        fm = char_matrix_to_featmat(data, '3d')

        n_permutations = 3
        alpha_sig = float(alphas[chrom])
        alpha_sig_EV = float(alphas_EV[chrom])
        # hp = pickle.load(open(os.path.join(FINAL_RESULTS_DIR,'hyperparams','chrom{}.p'.format(chrom)),'rb'))
        hp = best_params_montaez_2
        hp['n_snps'] = fm.shape[1]
        with tensorflow.Session().as_default():
            model = create_montaez_dense_model_2(hp)
            model.fit(x=fm[real_idx.train],
                      y=real_labels_cat[real_idx.train],
                      validation_data=(fm[real_idx.test], real_labels_cat[real_idx.test]),

                      epochs=hp['epochs'],
                      callbacks=[
                          ReduceLROnPlateau(monitor='val_loss',
                                            factor=hp['factor'],
                                            patience=hp['patience'],
                                            mode='min'
                                            ),
                      ],
                      verbose=1)

        t_star = permuted_deepcombi_method(model, hp, data, fm, real_labels, real_labels_cat, n_permutations, alpha_sig,
                                           real_pnorm_feature_scaling, filter_window_size, real_top_k, mode='min')
        t_star_EV = permuted_deepcombi_method(model, hp, data, fm, real_labels, real_labels_cat, n_permutations,
                                              alpha_sig_EV, real_pnorm_feature_scaling, filter_window_size, real_top_k,
                                              mode='all')
        pickle.dump(t_star, open(os.path.join(FINAL_RESULTS_DIR, 'chrom{}-t_star.p'.format(chrom)), 'wb'))
        pickle.dump(t_star_EV, open(os.path.join(FINAL_RESULTS_DIR, 'chrom{}-t_star_EV.p'.format(chrom)), 'wb'))

    def test_svm_accuracy(self, real_h5py_data, real_labels, real_labels_0based, real_idx):
        data = real_h5py_data(1)[:]
        x = char_matrix_to_featmat(data, '2d')
        print('Fitting data...')
        svm_model = real_classifier.fit(x[real_idx.train], real_labels[real_idx.train])
        print(svm_model.score(x[real_idx.test], real_labels[real_idx.test]))

    def test_parameters(self, real_h5py_data, real_labels_cat, real_idx):

        data = real_h5py_data(3)[:]
        fm = char_matrix_to_featmat(data, '3d')

        hp = pickle.load(open(os.path.join(FINAL_RESULTS_DIR, 'hyperparams', 'CD', 'chrom3.p'), 'rb'))
        hp['epochs'] = int(hp['epochs'])

        model = create_montaez_dense_model_2(hp)

        model.fit(x=fm[real_idx.train],
                  y=real_labels_cat[real_idx.train],
                  validation_data=(fm[real_idx.test], real_labels_cat[real_idx.test]),

                  epochs=hp['epochs'],
                  callbacks=[
                      TensorBoard(log_dir=os.path.join(FINAL_RESULTS_DIR, 'tb', 'tests'),
                                  histogram_freq=3,
                                  write_graph=False,
                                  write_grads=True,
                                  write_images=False)
                  ],
                  verbose=1)

    def test_extract_best_hps(self):
        for chromo in tqdm(range(1, 23)):

            for disease_id in disease_IDs:
                try:

                    dirname = os.path.join('talos_chrom_{}'.format(chromo))  # TODO CHANGE PREFIX

                    files = os.listdir(dirname)
                    r = talos.Reporting(os.path.join(dirname, files[0]))

                    data = r.table('val_acc', sort_by='val_acc', ascending=False)
                    best_row = data[data['acc'] > 0.80].iloc[0]
                    best_hps = best_row.to_dict()
                    best_hps['epochs'] = int(best_hps['epochs'])
                    best_hps['hidden_neurons'] = 6
                    best_hps['lr'] = 1e-4
                    best_hps['l1_reg'] = 1e-5
                    best_hps['n_snps'] = int(best_hps['n_snps'])

                    filename = os.path.join(FINAL_RESULTS_DIR, 'hyperparams', disease_id, 'chrom{}.p'.format(chromo))
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    pickle.dump(best_hps, open(filename, 'wb'))
                except Exception as identifier:
                    print('Failed for item {}. Reason:{}'.format(chromo, identifier))
                    raise ValueError(identifier)

    def test_train_models_with_best_params(self, real_h5py_data, real_labels_cat, real_idx):
        """ Generate a per-chromosom trained model for futur LRP-mapping quality assessment
        """
        chrom = int(os.environ['SGE_TASK_ID'])

        def train_model_on_disease(data_provider, disease_id, chrom):
            # Load data, hp & labels
            data = data_provider(disease_id, chrom)[:]

            fm = char_matrix_to_featmat(data, '3d')

            labels_cat = real_labels_cat(disease_id)

            hp = pickle.load(
                open(os.path.join(FINAL_RESULTS_DIR, 'hyperparams', disease_id, 'chrom{}.p'.format(chrom)), 'rb'))
            hp['epochs'] = 250  # int(hp['epochs']) TODO remove me

            idx = real_idx(disease_id)
            # Train 
            with tensorflow.Session().as_default():
                model = create_montaez_dense_model_2(hp)
                model.fit(x=fm[idx.train],
                          y=labels_cat[idx.train],
                          validation_data=(fm[idx.test], labels_cat[idx.test]),
                          epochs=hp['epochs'],
                          callbacks=[
                              TensorBoard(
                                  log_dir=os.path.join(FINAL_RESULTS_DIR, 'tb', disease_id, 'chrom{}'.format(chrom)),
                                  histogram_freq=10,
                                  write_graph=False,
                                  write_grads=False,
                                  write_images=False)
                          ],
                          verbose=0)
                filename = os.path.join(FINAL_RESULTS_DIR, 'trained_models', disease_id, 'model{}.h5'.format(chrom))
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                model.save(filename)

        Parallel(n_jobs=len(disease_IDs))(
            delayed(train_model_on_disease)(real_h5py_data, disease, chrom) for disease in tqdm(disease_IDs))

    def test_plot_all_pvalues(self, real_h5py_data, real_labels, alphas):

        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches(18.5, 10.5)
        axes.axhline(y=5)
        idx = 0
        for i in tqdm(range(1, 22)):
            h5py_data = real_h5py_data(i)[:]
            complete_pvalues = chi_square(h5py_data, real_labels)
            informative_idx = np.argwhere(complete_pvalues < 1e-5)
            color = np.zeros((len(complete_pvalues), 3))
            color[:] = [255 * (i % 2 == 0), 0, 255 * (i % 2 != 0)]
            color[informative_idx] = [0, 255, 0]

            axes.scatter(range(idx, idx + len(complete_pvalues)), -np.log10(complete_pvalues), c=color / 255.0,
                         marker='x')
            idx += len(complete_pvalues)
        fig.savefig(os.path.join(IMG_DIR, 'genome-wide-pvalues.png'))

    def test_generate_plots(self, real_labels, real_h5py_data):

        chromos = range(1, 23)
        offsets = np.zeros(len(chromos) + 1, dtype=np.int)
        middle_offset_history = np.zeros(len(chromos), dtype=np.int)

        # Populate offsets
        for i, chromo in enumerate(chromos):
            n_snps = real_h5py_data(chromo).shape[1]
            offsets[i + 1] = offsets[i] + n_snps

            middle_offset_history[i] = offsets[i] + int(n_snps / 2)

        for disease_id in tqdm(disease_IDs):

            labels = real_labels(disease_id)

            chrom_fig, axes = plt.subplots(4, 1, sharex='col')
            chrom_fig.set_size_inches(18.5, 10.5)

            raw_pvalues_ax = axes[0]
            c_selected_pvalues_ax = axes[1]
            pp_rm_ax = axes[2]
            deepc_selected_pvalues_ax = axes[3]

            def f(c_idx, chromo):

                # t_star = pickle.load(open(os.path.join(FINAL_RESULTS_DIR,disease,'chrom{}-t_star.p'.format(chrom)),'rb'))
                # t_star_EV = pickle.load(open(os.path.join(FINAL_RESULTS_DIR,disease,'chrom{}-t_star_EV.p'.format(chrom)),'rb'))
                data = real_h5py_data(chromo)[:]
                x_2d = char_matrix_to_featmat(data, '2d')
                x_3d = char_matrix_to_featmat(data, '3d')
                offset = offsets[c_idx]
                n_snps = x_3d.shape[1]

                # Generate LRP-based RAW- and postprocessed-RM
                model = load_model(
                    os.path.join(FINAL_RESULTS_DIR, 'trained_models', disease_id, 'model{}.h5'.format(chromo)))

                for i, layer in enumerate(model.layers):
                    if layer.name == 'dense_1':
                        layer.name = 'blu{}'.format(str(i))
                    if layer.name == 'dense_2':
                        layer.name = 'bla{}'.format(str(i))
                    if layer.name == 'dense_3':
                        layer.name = 'bleurg{}'.format(str(i))

                model = iutils.keras.graph.model_wo_softmax(model)
                analyzer = innvestigate.analyzer.LRPAlpha1Beta0(model)
                deepcombi_rm = np.absolute(analyzer.analyze(x_3d).sum(0))
                top_indices_deepcombi, pp_rm = postprocess_weights(deepcombi_rm, real_top_k, filter_window_size, p_svm,
                                                                   real_p_pnorm_filter)

                # Generate RAW- and postprocessed-PV
                raw_pvalues = chi_square(data, labels)

                if disease_id == 'CAD' and chromo != 9:
                    raw_pvalues[-np.log10(raw_pvalues) > 6] = 1

                top_indices_combi, _ = combi_method(data, x_2d, labels, real_pnorm_feature_scaling,
                                                    filter_window_size, real_top_k)

                combi_selected_pvalues = np.ones(n_snps)
                combi_selected_pvalues[top_indices_combi] = raw_pvalues[top_indices_combi]

                deepcombi_selected_pvalues = np.ones(n_snps)
                deepcombi_selected_pvalues[top_indices_deepcombi] = raw_pvalues[top_indices_deepcombi]

                # Plot stuff!

                informative_idx = np.argwhere(raw_pvalues < 1e-5)
                color = np.zeros((len(raw_pvalues), 3))
                color[:] = [1, 0, 0] if (chromo % 2 == 0) else [0, 0, 1]
                color[informative_idx] = [0, 1, 0]
                raw_pvalues_ax.scatter(range(offset, offset + n_snps), -np.log10(raw_pvalues), c=color, marker='x')
                deepc_selected_pvalues_ax.scatter(range(offset, offset + n_snps), -np.log10(deepcombi_selected_pvalues),
                                                  c=color, marker='x')
                c_selected_pvalues_ax.scatter(range(offset, offset + n_snps), -np.log10(combi_selected_pvalues),
                                              c=color, marker='x')
                pp_rm_ax.scatter(range(offset, offset + n_snps), pp_rm, c=color, marker='x')

            for c_idx, chromo in tqdm(enumerate(chromos)):
                f(c_idx, chromo)
            # Parallel(n_jobs=len(chromos), require='sharedmem')(delayed(f)(c_idx, chromo)

            plt.setp(raw_pvalues_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
            plt.setp(deepc_selected_pvalues_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
            plt.setp(c_selected_pvalues_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
            plt.setp(pp_rm_ax, xticks=middle_offset_history, xticklabels=range(1, 23))

            raw_pvalues_ax.set_title('Raw p-values')
            pp_rm_ax.set_title('DeepCOMBI-posprocessed')
            deepc_selected_pvalues_ax.set_title('PValues preselected by DeepCOMBI')
            c_selected_pvalues_ax.set_title('PValues preselected by COMBI')

            deepc_selected_pvalues_ax.set_xlabel('Chromosome')
            pp_rm_ax.set_ylim(bottom=0.0)

            chrom_fig.savefig(os.path.join(FINAL_RESULTS_DIR, 'plots', '{}-manhattan.png'.format(disease_id)))

