import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import pickle
import numpy as np
import tensorflow
from models import create_montaez_dense_model_2, best_params_montaez_2
from keras.callbacks import TensorBoard, ReduceLROnPlateau, CSVLogger
from helpers import char_matrix_to_featmat, postprocess_weights, get_available_gpus
from parameters_complete import FINAL_RESULTS_DIR, IMG_DIR, real_pnorm_feature_scaling, real_p_pnorm_filter, filter_window_size, real_top_k, p_svm
from parameters_complete import disease_IDs

from keras.models import load_model
from keras import backend as K
from combi import permuted_deepcombi_method, combi_method, real_classifier
from joblib import Parallel, delayed
from tqdm import tqdm
import talos
from talos.utils.gpu_utils import parallel_gpu_jobs

import innvestigate
import innvestigate.utils as iutils

TEST_PERCENTAGE = 0.20


class TestLOTR(object):

    def test_lrp(self, real_h5py_data, real_labels, real_labels_0based,real_pvalues, real_labels_cat, real_idx, rep, tmp_path):

        fig, axes = plt.subplots(5, 1, sharex='col')

        h5py_data = real_h5py_data(1)

        x_2d = char_matrix_to_featmat(h5py_data, '2d', real_pnorm_feature_scaling)
        x_3d = char_matrix_to_featmat(h5py_data, '3d', real_pnorm_feature_scaling)

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

        complete_pvalues = real_pvalues('CD',1)
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
            data = real_h5py_data(chrom)
            x_2d = char_matrix_to_featmat(data, '2d', real_pnorm_feature_scaling)

            real_classifier.fit(x_2d[real_idx.train], real_labels[real_idx.train])
            score += real_classifier.score(x_2d[real_idx.test], real_labels[real_idx.test])
        score /= len(chroms)
        print("Average SVM test accuracy : {}".format(score))

    def test_class_proportions(self, real_labels, real_idx):
        train_labels = real_labels[real_idx.train]
        test_labels = real_labels[real_idx.test]

        print("Train: Cases:{};  Controls:{}".format((train_labels > 0.5).sum(), (train_labels < 0.5).sum()))
        print("Test: Cases:{};  Controls:{}".format((test_labels > 0.5).sum(), (test_labels < 0.5).sum()))



    def test_hpsearch(self, real_h5py_data, real_labels_cat, real_idx):
        """ Runs HP search for a subset of chromosomes
        """
        # Each node gets a set of chromosomes to process :D
        
        disease = disease_IDs[int(os.environ['SGE_TASK_ID'])-1]

        for chrom in range(1,23):

            # 1. Do hyperparam search on each chromosome and find parameters with BEST VAL ACCURAC

            data = real_h5py_data(disease, chrom)
            fm = char_matrix_to_featmat(data, '3d',real_pnorm_feature_scaling)
            labels_cat = real_labels_cat(disease)
            idx = real_idx(disease)
            params_space = {
                'n_snps': [fm.shape[1]],
                'epochs': [400],
                'dropout_rate': [0.3],
                'l1_reg': list(np.logspace(-6, -2, 5)),
                'l2_reg': [0],
                'hidden_neurons': [3, 6, 10],
                'lr': list(np.logspace(-4, -2, 3)),
            }

            def talos_wrapper(x, y, x_val, y_val, params):
                model = create_montaez_dense_model_2(params)
                out = model.fit(x=x,
                                y=y,
                                validation_data=(x_val, y_val),
                                epochs=params['epochs'],
                                verbose=0)
                return out, model

            nb_gpus = get_available_gpus()

            if nb_gpus == 1:
                parallel_gpu_jobs(0.33)

            os.makedirs(os.path.join(FINAL_RESULTS_DIR,'talos',disease,str(chrom)), exist_ok=True)

            talos.Scan(x=fm[idx.train],
                    y=labels_cat[idx.train],
                    x_val=fm[idx.test],
                    y_val=labels_cat[idx.test],
                    reduction_method='gamify',
                    # reduction_interval=10,
                    # reduction_window=10,
                    # reduction_metric='val_acc',
                    # reduction_threshold=0.2,
                    minimize_loss=False,
                    params=params_space,
                    model=talos_wrapper,
                    experiment_name='final_results/talos/'+disease+'/'+str(chrom))

    def test_permutations(self, real_h5py_data, real_labels, real_labels_0based, real_labels_cat, real_idx, alphas,
                          alphas_EV):
        """ Computes t_star for each chromosome thanks to the permutation method.
        """

        chrom = int(os.environ['SGE_TASK_ID'])

        # 1. Do hyperparam search on each chromosome and find parameters with BEST VAL ACCURAC

        data = real_h5py_data(chrom)

        fm = char_matrix_to_featmat(data, '3d', real_pnorm_feature_scaling)

        n_permutations = 3
        alpha_sig = float(alphas[chrom])
        alpha_sig_EV = float(alphas_EV[chrom])
        # hp = pickle.load(open(os.path.join(FINAL_RESULTS_DIR,'hyperparams','chrom{}.p'.format(chrom)),'rb'))
        hp = best_params_montaez_2
        hp['n_snps'] = fm.shape[1]
        with tensorflow.compat.v1.Session().as_default():
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
                                            filter_window_size, real_top_k, mode='min')
        t_star_EV = permuted_deepcombi_method(model, hp, data, fm, real_labels, real_labels_cat, n_permutations,
                                              alpha_sig_EV, filter_window_size, real_top_k,
                                              mode='all')
        pickle.dump(t_star, open(os.path.join(FINAL_RESULTS_DIR, 'chrom{}-t_star.p'.format(chrom)), 'wb'))
        pickle.dump(t_star_EV, open(os.path.join(FINAL_RESULTS_DIR, 'chrom{}-t_star_EV.p'.format(chrom)), 'wb'))

    def test_svm_accuracy(self, real_h5py_data, real_labels, real_labels_0based, real_idx):
        data = real_h5py_data(1)
        x = char_matrix_to_featmat(data, '2d',real_pnorm_feature_scaling)
        print('Fitting data...')
        svm_model = real_classifier.fit(x[real_idx.train], real_labels[real_idx.train])
        print(svm_model.score(x[real_idx.test], real_labels[real_idx.test]))

    def test_parameters(self, real_h5py_data, real_labels_cat, real_idx):

        data = real_h5py_data(3)
        fm = char_matrix_to_featmat(data, '3d', real_pnorm_feature_scaling)

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

                    dirname = os.path.join('talos_CD_l1_lr_hn','talos_chrom_{}'.format(chromo))  # TODO CHANGE PREFIX

                    files = os.listdir(dirname)
                    r = talos.Reporting(os.path.join(dirname, files[0]))

                    data = r.table('val_acc', sort_by='val_acc', ascending=False)
                    best_row = data[data['acc'] > 0.80].iloc[0]
                    best_hps = best_row.to_dict()
                    best_hps['epochs'] = 250
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

        for disease_id in ['T1D','T2D']:#tqdm(disease_IDs):

            # Load data, hp & labels
            data = real_h5py_data(disease_id, chrom)
            fm = char_matrix_to_featmat(data, '3d', real_pnorm_feature_scaling)

            labels_cat = real_labels_cat(disease_id)

            hp = pickle.load(open(os.path.join(FINAL_RESULTS_DIR, 'hyperparams', disease_id, 'chrom{}.p'.format(chrom)), 'rb'))
            hp['epochs'] = int(hp['epochs']) #TODO remove me
            print(hp)

            idx = real_idx(disease_id)
            # Train 

            model = create_montaez_dense_model_2(hp)
            model.fit(x=fm[idx.train],
                        y=labels_cat[idx.train],
                        validation_data=(fm[idx.test], labels_cat[idx.test]),
                        epochs=hp['epochs'],
                        callbacks=[
                            CSVLogger(os.path.join(FINAL_RESULTS_DIR, 'csv_logs', disease_id, '{}'.format(chrom)))
                        ],
                        verbose=0)
            filename = os.path.join(FINAL_RESULTS_DIR, 'trained_models', disease_id, 'model{}.h5'.format(chrom))
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            model.save(filename)
            K.clear_session()
            del data, fm, model 


    def test_generate_plots(self, real_pvalues):
        """ Plot manhattan figures of rpvt VS deepcombi, for one specific disease 
        SGE_TASK_ID-1 : index of the disease processed by this node
        """
        disease_id = disease_IDs[int(os.environ['SGE_TASK_ID'])-1]

        chromos = range(1, 23)
        offsets = np.zeros(len(chromos) + 1, dtype=np.int)
        middle_offset_history = np.zeros(len(chromos), dtype=np.int)


        chrom_fig, axes = plt.subplots(3, 1, sharex='col')
        chrom_fig.set_size_inches(18.5, 10.5)

        raw_pvalues_ax = axes[0]
        c_selected_pvalues_ax = axes[1]
        deepc_selected_pvalues_ax = axes[2]

        top_indices_deepcombi = np.load(
            os.path.join(FINAL_RESULTS_DIR, 'deepcombi_selected_indices', '{}.npy'.format(disease_id)))
        top_indices_combi = np.load(
            os.path.join(FINAL_RESULTS_DIR, 'combi_selected_indices', '{}.npy'.format(disease_id)))

        complete_pvalues = []
        n_snps = np.zeros(22)
        for i,chromo in enumerate(chromos):

            raw_pvalues = real_pvalues(disease_id, chromo)


            if disease_id == 'CAD' and chromo != 9:
                raw_pvalues[raw_pvalues < 1e-6] = 1

            complete_pvalues += raw_pvalues.tolist()

            n_snps[i] = len(raw_pvalues)
            offsets[i + 1] = offsets[i] + n_snps[i]
            middle_offset_history[i] = offsets[i] + int(n_snps[i] / 2)

        complete_pvalues = np.array(complete_pvalues).flatten()


        combi_selected_pvalues = np.ones(len(complete_pvalues))
        combi_selected_pvalues[top_indices_combi] = complete_pvalues[top_indices_combi]

        deepcombi_selected_pvalues = np.ones(len(complete_pvalues))
        deepcombi_selected_pvalues[top_indices_deepcombi] = complete_pvalues[top_indices_deepcombi]


        color = np.zeros((len(complete_pvalues), 3))
        informative_idx = np.argwhere(complete_pvalues < 1e-5)

        alt = True
        for i,offset in enumerate(offsets[:-1]):
            color[offset:offsets[i+1]] = [1, 0, 0] if alt else [0, 0, 1]
            alt = not alt


        color[informative_idx] = [0, 1, 0]
        raw_pvalues_ax.scatter(range(len(complete_pvalues)),-np.log10(complete_pvalues), c=color, marker='x')
        deepc_selected_pvalues_ax.scatter( range(len(complete_pvalues)),-np.log10(deepcombi_selected_pvalues),
                                            c=color, marker='x')
        c_selected_pvalues_ax.scatter(range(len(complete_pvalues)),-np.log10(combi_selected_pvalues),
                                        c=color, marker='x')



        plt.setp(raw_pvalues_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
        plt.setp(deepc_selected_pvalues_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
        plt.setp(c_selected_pvalues_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
        chrom_fig.canvas.set_window_title(disease_id)
        raw_pvalues_ax.set_title('Raw p-values')
        deepc_selected_pvalues_ax.set_title('PValues preselected by DeepCOMBI')
        c_selected_pvalues_ax.set_title('PValues preselected by COMBI')

        deepc_selected_pvalues_ax.set_xlabel('Chromosome')

        chrom_fig.savefig(os.path.join(FINAL_RESULTS_DIR, 'plots', '{}-manhattan.png'.format(disease_id)))

