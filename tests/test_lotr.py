import matplotlib

matplotlib.use('Agg')

import os
import pandas as pd
import pickle
import numpy as np
import tensorflow
from models import create_montaez_dense_model, best_params_montaez
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from helpers import char_matrix_to_featmat, get_available_gpus
from parameters_complete import FINAL_RESULTS_DIR, real_pnorm_feature_scaling, filter_window_size, real_top_k
from parameters_complete import disease_IDs

from keras import backend as K
from combi import permuted_deepcombi_method, real_classifier
from tqdm import tqdm
import talos
from talos.utils.gpu_utils import parallel_gpu_jobs

TEST_PERCENTAGE = 0.20


class TestLOTR(object):


    def test_class_proportions(self, real_labels, real_idx):
        for disease in disease_IDs:
            labels = real_labels(disease)
            idx = real_idx()
            train_labels = labels[real_idx.train]
            test_labels = labels[real_idx.test]
            print("Disease: {}".format(disease))
            print("Train: Cases:{};  Controls:{}".format((train_labels > 0.5).sum(), (train_labels < 0.5).sum()))
            print("Test: Cases:{};  Controls:{}".format((test_labels > 0.5).sum(), (test_labels < 0.5).sum()))



    def test_hpsearch(self, real_genomic_data, real_labels_cat, real_idx):
        """ Runs HP search for a subset of chromosomes
        """
        # Each node gets a set of chromosomes to process :D
        
        disease = disease_IDs[int(os.environ['SGE_TASK_ID'])-1]

        for chrom in range(1,23):

            # 1. Do hyperparam search on each chromosome and find parameters with BEST VAL ACCURAC

            data = real_genomic_data(disease, chrom)
            fm = char_matrix_to_featmat(data, '3d',real_pnorm_feature_scaling)
            labels_cat = real_labels_cat(disease)
            idx = real_idx(disease)
            params_space = {
                'n_snps': [fm.shape[1]],
                'epochs': [600],
                'dropout_rate': [0.3],
                'l1_reg': list(np.logspace(-6, -2, 5)),
                'l2_reg': [0],
                'hidden_neurons': [3, 6, 10],
                'lr': list(np.logspace(-4, -2, 3)),
            }

            def talos_wrapper(x, y, x_val, y_val, params):
                model = create_montaez_dense_model(params)
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
                    #reduction_method='gamify',
                    # reduction_interval=10,
                    # reduction_window=10,
                    # reduction_metric='val_acc',
                    # reduction_threshold=0.2,
                    minimize_loss=False,
                    params=params_space,
                    model=talos_wrapper,
                    experiment_name='final_results/talos/'+ disease + '/'+str(chrom))

    def test_permutations(self, real_genomic_data, real_labels, real_labels_0based, real_labels_cat, real_idx, alphas,
                          alphas_ev):
        """ Computes t_star for each chromosome thanks to the permutation method.
        """

        chrom = int(os.environ['SGE_TASK_ID'])

        # 1. Do hyperparam search on each chromosome and find parameters with BEST VAL ACCURAC

        data = real_genomic_data('CD', chrom)

        fm = char_matrix_to_featmat(data, '3d', real_pnorm_feature_scaling)

        n_permutations = 3
        alpha_sig = float(alphas[chrom])
        alpha_sig_EV = float(alphas_ev[chrom])
        # hp = pickle.load(open(os.path.join(FINAL_RESULTS_DIR,'hyperparams','chrom{}.p'.format(chrom)),'rb'))
        hp = best_params_montaez
        hp['n_snps'] = fm.shape[1]
        with tensorflow.compat.v1.Session().as_default():
            model = create_montaez_dense_model(hp)
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

        t_star = permuted_deepcombi_method(real_classifier, model, hp, data, fm, real_labels, real_labels_cat, n_permutations, alpha_sig,
                                            filter_window_size, real_top_k, mode='min')
        t_star_EV = permuted_deepcombi_method(model, hp, data, fm, real_labels, real_labels_cat, n_permutations,
                                              alpha_sig_EV, filter_window_size, real_top_k,
                                              mode='all')
        pickle.dump(t_star, open(os.path.join(FINAL_RESULTS_DIR, 'chrom{}-t_star.p'.format(chrom)), 'wb'))
        pickle.dump(t_star_EV, open(os.path.join(FINAL_RESULTS_DIR, 'chrom{}-t_star_EV.p'.format(chrom)), 'wb'))



    def test_extract_best_hps(self):

        for disease_id in disease_IDs:
            try:
                data = pd.DataFrame()
                talos_disease_directory = os.path.join(FINAL_RESULTS_DIR, 'talos', disease_id)
                for root, _, files in os.walk(talos_disease_directory):
                    for file in files:
                        if file.endswith(".csv"):
                            df = pd.read_csv(os.path.join(root, file))
                            data = data.append(df, ignore_index=True)

                data.sort_values(by=['val_acc'], ascending=False, inplace=True)
                best_hps = data[data['acc'] > 0.80].iloc[0].to_dict()
                #best_hps['epochs'] = 250
                #best_hps['hidden_neurons'] = 6
                #best_hps['lr'] = 1e-4
                #best_hps['l1_reg'] = 1e-5

                for chromo in tqdm(range(1, 23)):
                    filename = os.path.join(FINAL_RESULTS_DIR, 'hyperparams', disease_id, 'chrom{}.p'.format(chromo))
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    pickle.dump(best_hps, open(filename, 'wb'))
            except Exception as identifier:
                print('Failed for item {}. Reason:{}'.format(disease_id, identifier))
                raise ValueError(identifier)


    def test_train_models_with_best_params(self, real_genomic_data, real_labels_cat, real_idx):
        """ Generate a per-chromosom trained model for futur LRP-mapping quality assessment
        TRAINS ON WHOLE DATASET
        """
        candidates = []
        for disease_id in disease_IDs:
            for chrom in range(1,23):
                candidates.append([disease_id, chrom])
        
        idx = int(os.environ['SGE_TASK_ID'])-1
        disease_id, chrom = candidates[idx]


        # Load data, hp & labels
        data = real_genomic_data(disease_id, chrom)
        fm = char_matrix_to_featmat(data, '3d', real_pnorm_feature_scaling)

        labels_cat = real_labels_cat(disease_id)

        hp = pickle.load(open(os.path.join(FINAL_RESULTS_DIR, 'hyperparams', disease_id, 'chrom{}.p'.format(chrom)), 'rb'))
        hp['epochs'] = int(hp['epochs'])
        hp['n_snps'] = int(fm.shape[1])
        hp['epochs'] = 250
        hp['hidden_neurons'] = 6
        hp['lr'] = 1e-4
        hp['l1_reg'] = 1e-5 # TODO change me back
        # Train
        os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'csv_logs', disease_id), exist_ok=True)

        model = create_montaez_dense_model(hp)
        model.fit(x=fm,
                    y=labels_cat,
                    epochs=hp['epochs'],
                    callbacks=[
                        #CSVLogger(os.path.join(FINAL_RESULTS_DIR, 'csv_logs', disease_id, '{}'.format(chrom)))
                    ],
                    verbose=0)
        filename = os.path.join(FINAL_RESULTS_DIR, 'trained_models', disease_id, 'model{}.h5'.format(chrom))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        model.save(filename)
        K.clear_session()
        del data, fm, model

    def test_hpsearch_crohn(self, real_genomic_data, real_labels_cat, real_idx):
        """ Runs HP search for a subset of chromosomes
        """

        disease = 'CD'  # disease_IDs[int(os.environ['SGE_TASK_ID'])-1]

        for chrom in [3]:  # range(1,23):


            data = real_genomic_data(disease, chrom)
            fm = char_matrix_to_featmat(data, '3d', real_pnorm_feature_scaling)
            labels_cat = real_labels_cat(disease)
            idx = real_idx(disease)
            params_space = {
                'n_snps': [fm.shape[1]],
                'epochs': [600],
                'dropout_rate': [0.3],
                'l1_reg': list(np.logspace(-6, -2, 5)),
                'l2_reg': [0],
                'hidden_neurons': [3, 6, 10],
                'lr': list(np.logspace(-4, -2, 3)),
            }

            def talos_wrapper(x, y, x_val, y_val, params):
                model = create_montaez_dense_model(params)
                out = model.fit(x=x,
                                y=y,
                                validation_data=(x_val, y_val),
                                epochs=params['epochs'],
                                verbose=0)
                return out, model

            nb_gpus = get_available_gpus()

            if nb_gpus == 1:
                parallel_gpu_jobs(0.33)

            os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'talos', disease, str(chrom)), exist_ok=True)

            talos.Scan(x=fm[idx.train],
                       y=labels_cat[idx.train],
                       x_val=fm[idx.test],
                       y_val=labels_cat[idx.test],
                       reduction_method='gamify',
                       minimize_loss=False,
                       params=params_space,
                       model=talos_wrapper,
                       experiment_name=os.path.join(FINAL_RESULTS_DIR,'talos',disease, str(chrom)))


    def test_crohn_c3_parameters(self, real_genomic_data, real_labels_cat, real_idx):

        data = real_genomic_data('CD', 3)
        fm = char_matrix_to_featmat(data, '3d', real_pnorm_feature_scaling)

        hp = dict(epochs=600,dropout_rate=0.3,hidden_neurons=10,l1_reg=0.0001,l2_reg=0,lr=0.01,n_snps=fm.shape[1])

        model = create_montaez_dense_model(hp)

        model.fit(x=fm[real_idx.train],
                  y=real_labels_cat[real_idx.train],
                  validation_data=(fm[real_idx.test], real_labels_cat[real_idx.test]),

                  epochs=hp['epochs'],
                  callbacks=[
                      CSVLogger(os.path.join(FINAL_RESULTS_DIR, 'csv_logs', 'CD', '3'))
                  ],
                  verbose=1)