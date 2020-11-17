import matplotlib

matplotlib.use('Agg')

import os
import pandas as pd
import pickle
import numpy as np
import tensorflow
from models import create_montaez_dense_model, best_params_montaez
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from helpers import char_matrix_to_featmat, get_available_gpus, chi_square
from parameters_complete import FINAL_RESULTS_DIR, real_pnorm_feature_scaling, filter_window_size_deep, real_p_pnorm_filter_deep, p_svm_deep, real_top_k_deep, pvalue_threshold, n_permutations, filter_window_size, real_p_pnorm_filter, p_svm, real_top_k 

from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, accuracy_score
from sklearn.utils import class_weight
from sklearn.svm import LinearSVC

from parameters_complete import disease_IDs

from keras import backend as K
from combi import permuted_deepcombi_method,permuted_combi_method, real_classifier
from tqdm import tqdm
import talos
from talos.utils.gpu_utils import parallel_gpu_jobs
import pdb

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
        #disease = disease_IDs[int(1)-1]

        for chrom in range(1,22):

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
                    # reduction_method='gamify',
                    # reduction_interval=10,
                    # reduction_window=10,
                    # reduction_metric='val_acc',
                    # reduction_threshold=0.2,
                    # round_limit = 100,
                    minimize_loss=False,
                    params=params_space,
                    model=talos_wrapper,
                    experiment_name='MONTAEZ/talos/'+ disease + '/'+str(chrom))
                    #experiment_name=os.path.join('experiments','MONTAEZ_like_Alex','talos',disease,str(chrom))
                    #experiment_name=os.path.join(FINAL_RESULTS_DIR,'talos',disease,str(chrom)))


    def test_hpsearch_with_weights(self, real_genomic_data, real_labels_cat, real_idx):
        """ Runs HP search for a subset of chromosomes
        """
        # Each node gets a set of chromosomes to process :D
        disease = disease_IDs[int(os.environ['SGE_TASK_ID'])-1]
        #disease = disease_IDs[int(1)-1]

        for chrom in range(1,22):

            # 1. Do hyperparam search on each chromosome and find parameters with BEST VAL ACCURAC

            data = real_genomic_data(disease, chrom)
            fm = char_matrix_to_featmat(data, '3d',real_pnorm_feature_scaling)
            labels_cat = real_labels_cat(disease)
            
            # Do centering and scaling like Marina
            fm[fm>0]=1
            fm[fm<0]=0.

            mean = np.mean(np.mean(fm,0),0)
            std = np.std(np.std(fm,0),0)

            fm = (fm-mean)/std

            idx = real_idx(disease)
            params_space = {
                'n_snps': [fm.shape[1]],
                'epochs': [10,50,100],
                'dropout_rate': [0.3],
                'l1_reg': [0.1,0.01,0.001],
                'l2_reg': [0, 0.0001],
                'hidden_neurons': [64],
                'lr': [0.0001, 0.00001, 0.000001],
            }

            def talos_wrapper(x, y, x_val, y_val, params):
                model = create_montaez_dense_model(params)
                y_integers = np.argmax(y, axis=1)
                class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
                d_class_weights = dict(enumerate(class_weights))
                out = model.fit(x=x,
                                y=y,
                                validation_data=(x_val, y_val),
                                epochs=params['epochs'],
                                verbose=0,
                                class_weight=d_class_weights)
                return out, model

            nb_gpus = get_available_gpus()

            if nb_gpus == 1:
                parallel_gpu_jobs(0.33)

            os.makedirs(os.path.join(FINAL_RESULTS_DIR,'talos',disease,str(chrom)), exist_ok=True)

            talos.Scan(x=fm[idx.train],
                    y=labels_cat[idx.train],
                    x_val=fm[idx.test],
                    y_val=labels_cat[idx.test],
                    # reduction_method='gamify',
                    # reduction_interval=10,
                    # reduction_window=10,
                    # reduction_metric='val_acc',
                    # reduction_threshold=0.2,
                    # round_limit = 100,
                    # minimize_loss=False,
                    params=params_space,
                    model=talos_wrapper,
                    experiment_name='MONTAEZ_with_classweights_and_new_scaling/talos/'+ disease + '/'+str(chrom))
                    #experiment_name=os.path.join('experiments','MONTAEZ_like_Alex','talos',disease,str(chrom))
                    #experiment_name=os.path.join('MONTAEZ_with_classweights_and_new_scaling','talos',disease,str(chrom)))


    def test_permutations(self, real_genomic_data, real_labels, real_labels_0based, real_labels_cat, real_idx, alphas, alphas_ev):
        """ Computes t_star for each chromosome thanks to the permutation method. """
        candidates = []
        for disease_id in disease_IDs:
            for chrom in range(1,23):
                candidates.append([disease_id, chrom])

        idx = int(os.environ['SGE_TASK_ID'])-1
        disease_id, chrom = candidates[idx]
        print(disease_id)
        print(chrom)

        # Load data, hp & labels
        data = real_genomic_data(disease_id, chrom)
        labels = real_labels(disease_id)
        alpha_sig = float(alphas(disease_id)[chrom-1])
        #alpha_sig_EV = float(alphas_ev[chrom-1])
        counter = 0
        while np.isnan(alpha_sig):
            alpha_sig= float(alphas(disease_id)[chrom + counter])
            counter += 1
        t_star = permuted_deepcombi_method(data, labels,disease_id, chrom,pvalue_threshold,real_pnorm_feature_scaling, filter_window_size_deep, real_p_pnorm_filter_deep, p_svm_deep, real_top_k_deep, n_permutations, alpha_sig, mode='min')

        #t_star_EV = permuted_deepcombi_method(data, labels, filter_window_size_deep, real_p_pnorm_filter_deep, p_svm_deep, real_top_k_deep, n_permutations, alpha_sig,  mode='all')
        os.makedirs(os.path.join(FINAL_RESULTS_DIR,'permtest_results',disease_id), exist_ok=True)

        pickle.dump(t_star, open(os.path.join(FINAL_RESULTS_DIR,'permtest_results',disease_id, 'chrom{}-t_star.p'.format(chrom)), 'wb'))
        #pickle.dump(t_star_EV, open(os.path.join(FINAL_RESULTS_DIR, 'chrom{}-t_star_EV.p'.format(chrom)), 'wb'))
		
        t_star_combi = permuted_combi_method(data, labels,real_pnorm_feature_scaling, filter_window_size, real_p_pnorm_filter, p_svm, real_top_k, n_permutations, alpha_sig, mode='min')
        pickle.dump(t_star_combi, open(os.path.join(FINAL_RESULTS_DIR,'permtest_results',disease_id, 'chrom{}-t_star_combi.p'.format(chrom)), 'wb'))
		
    def test_permutations_onlycombi(self, real_genomic_data, real_labels, real_labels_0based, real_labels_cat, real_idx, alphas, alphas_ev):
        """ Computes t_star for each chromosome thanks to the permutation method. """
        candidates = []
        for disease_id in disease_IDs:
            for chrom in range(1,23):
                candidates.append([disease_id, chrom])

        idx = int(os.environ['SGE_TASK_ID'])-1
        disease_id, chrom = candidates[idx]
        print(disease_id)
        print(chrom)

        # Load data, hp & labels
        data = real_genomic_data(disease_id, chrom)
        labels = real_labels(disease_id)
        alpha_sig = float(alphas(disease_id)[chrom-1])
        counter = 0

        while np.isnan(alpha_sig):
            alpha_sig= float(alphas(disease_id)[chrom + counter])
            counter += 1
        #alpha_sig_EV = float(alphas_ev[chrom-1])

        #t_star = permuted_deepcombi_method(data, labels,disease_id, chrom,pvalue_threshold,real_pnorm_feature_scaling, filter_window_size_deep, real_p_pnorm_filter_deep, p_svm_deep, real_top_k_deep, n_permutations, alpha_sig, mode='min')
		
        #t_star_EV = permuted_deepcombi_method(data, labels, filter_window_size_deep, real_p_pnorm_filter_deep, p_svm_deep, real_top_k_deep, n_permutations, alpha_sig,  mode='all')
        os.makedirs(os.path.join(FINAL_RESULTS_DIR,'permtest_results',disease_id), exist_ok=True)

        #pickle.dump(t_star, open(os.path.join(FINAL_RESULTS_DIR,'permtest_results',disease_id, 'chrom{}-t_star.p'.format(chrom)), 'wb'))
        #pickle.dump(t_star_EV, open(os.path.join(FINAL_RESULTS_DIR, 'chrom{}-t_star_EV.p'.format(chrom)), 'wb'))
        t_star_combi = permuted_combi_method(data, labels,real_pnorm_feature_scaling, filter_window_size, real_p_pnorm_filter, p_svm, real_top_k, n_permutations, alpha_sig, mode='min')
        pickle.dump(t_star_combi, open(os.path.join(FINAL_RESULTS_DIR,'permtest_results',disease_id, 'chrom{}-t_star_combi.p'.format(chrom)), 'wb'))



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
		
    def test_train_models_with_const_params(self, real_genomic_data,real_labels, real_labels_cat, real_idx):
        """ Generate a per-chromosom trained model for futur LRP-mapping quality assessment
        TRAINS ON WHOLE DATASET
        """
        candidates = []
        for disease_id in disease_IDs:
            for chrom in range(1,23):
                candidates.append([disease_id, chrom])
        
        idx = int(os.environ['SGE_TASK_ID'])-1

        disease_id, chrom = candidates[idx]
        print(disease_id)
        print(chrom)

        # Load data & labels
        data = real_genomic_data(disease_id, chrom)
        labels_cat = real_labels_cat(disease_id)
        labels = real_labels(disease_id)
       
        # pvalues
        rpvt_pvalues = chi_square(data, labels)

        # pvalue thresholding
        valid_snps = rpvt_pvalues < pvalue_threshold
        data = data[:,valid_snps,:]
        
        # Centering and Scaling
        fm = char_matrix_to_featmat(data, '3d', real_pnorm_feature_scaling)
        fm[fm>0]=1
        fm[fm<0]=0.
        mean = np.mean(np.mean(fm,0),0)
        std = np.std(np.std(fm,0),0)
        fm = (fm-mean)/std
        
        hp =  {'dropout_rate': 0.3, 'epochs': 500, 'hidden_neurons': 64, 'l1_reg': 0.001, 'l2_reg': 0.0001, 'lr': 1e-05, 'n_snps': int(fm.shape[1])} 
        # Optimal parameters for pthresh = 1e-02 
        # hp =  {'dropout_rate': 0.3, 'epochs': 500, 'hidden_neurons': 64, 'l1_reg': 0.001, 'l2_reg': 0.0001, 'lr': 1e-05, 'n_snps': int(fm.shape[1])} 

        # Create Model
        os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'csv_logs', disease_id), exist_ok=True)
        model = create_montaez_dense_model(hp)

        # Class weights
        y_integers = np.argmax(labels_cat, axis=1)
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
        d_class_weights = dict(enumerate(class_weights))

        # Train
        model.fit(x=fm, y=labels_cat, epochs=hp['epochs'], callbacks=[CSVLogger(os.path.join(FINAL_RESULTS_DIR, 'csv_logs', disease_id, '{}'.format(chrom)))], verbose=0, class_weight=d_class_weights)
		
        # Save
        filename = os.path.join(FINAL_RESULTS_DIR, 'trained_models', disease_id, 'model{}.h5'.format(chrom))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        model.save(filename)
        K.clear_session()
        del data, fm, model


    def test_train_models_save_accuracies(self, real_genomic_data,real_labels, real_labels_cat, real_idx):
        """ Generate a per-chromosom trained model and svm to save validation accuracies
        """
        for disease_id in disease_IDs:
            if disease_id == 'CD':
                continue
            auc_scores = []
            prec_scores = []
            acc_scores =[]
            bal_acc_scores =[]
            auc_scores_svm = []
            prec_scores_svm = []
            acc_scores_svm =[]
            bal_acc_scores_svm =[]
            for chrom in range(1,23):
                print(disease_id)
                print(chrom)

                # Load data, hp & labels
                data_orig = real_genomic_data(disease_id, chrom)
                data = data_orig
                labels_cat = real_labels_cat(disease_id)
                labels = real_labels(disease_id)
                
                # pvalues
                rpvt_pvalues = chi_square(data, labels)

                # pvalue thresholding
                valid_snps = rpvt_pvalues < pvalue_threshold
                data = data[:,valid_snps,:]
                
                # Centering and Scaling
                fm = char_matrix_to_featmat(data, '3d', real_pnorm_feature_scaling)
                fm[fm>0]=1
                fm[fm<0]=0.
                mean = np.mean(np.mean(fm,0),0)
                std = np.std(np.std(fm,0),0)
                fm = (fm-mean)/std
                
                idx_tt = real_idx(disease_id)
                X_train = fm[idx_tt.train]
                X_test = fm[idx_tt.test]

                hp =  {'dropout_rate': 0.3, 'epochs': 500, 'hidden_neurons': 64, 'l1_reg': 0.001, 'l2_reg': 0.0001, 'lr': 1e-05, 'n_snps': int(fm.shape[1])} 

                # Create Model
                model = create_montaez_dense_model(hp)

                # Class weights
                y_integers = np.argmax(labels_cat, axis=1)
                class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
                d_class_weights = dict(enumerate(class_weights))

                # Train
                model.fit(x=X_train, y=labels_cat[idx_tt.train], epochs=hp['epochs'], verbose=0,class_weight=d_class_weights)
                y_pred = model.predict(x=X_test)

                auc_scores.append(roc_auc_score(labels_cat[idx_tt.test], y_pred, average='weighted'))
                prec_scores.append(average_precision_score(labels_cat[idx_tt.test], y_pred, average='weighted'))
                acc_scores.append(accuracy_score(labels_cat[idx_tt.test][:,0], y_pred[:,0]>y_pred[:,1]))
                bal_acc_scores.append(balanced_accuracy_score(labels_cat[idx_tt.test][:,0], y_pred[:,0]>y_pred[:,1]))
                
                K.clear_session()
                del data, fm, model
                
                # Train SVM for comparison on original data (no pthresh)
                fm_svm = char_matrix_to_featmat(data_orig, '3d', 6)	
                #fm_svm[fm_svm>0]=1
                #fm_svm[fm_svm<0]=0.

                #mean = np.mean(np.mean(fm_svm,0),0)
                #std = np.std(np.std(fm_svm,0),0)

                #fm_svm = (fm_svm-mean)/std
                X_train_svm = fm_svm[idx_tt.train]
                X_test_svm = fm_svm[idx_tt.test]

                # Best model on original scaling
                clf = LinearSVC(penalty='l2', loss='hinge', C=1.0000e-05, dual=True, tol=1e-3, verbose=0)

                # Best on Marinas scaling
                #clf = LinearSVC(penalty='l1', C=.01, dual=False, tol=1e-1,fit_intercept=True, class_weight=None, verbose=0, random_state=0, max_iter=100)
                clf.fit(X=X_train_svm.reshape(len(X_train_svm),-1), y=labels_cat[idx_tt.train][:,0])
                y_pred_svm = clf.predict(X=X_test_svm.reshape(len(X_test_svm),-1))

                # Evaluate
                auc_scores_svm.append(roc_auc_score(labels_cat[idx_tt.test][:,0], y_pred_svm, average='weighted'))
                prec_scores_svm.append(average_precision_score(labels_cat[idx_tt.test][:,0], y_pred_svm, average='weighted'))
                acc_scores_svm.append(accuracy_score(labels_cat[idx_tt.test][:,0], y_pred_svm))
                bal_acc_scores_svm.append(balanced_accuracy_score(labels_cat[idx_tt.test][:,0], y_pred_svm))
                
            # Save

            os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease_id), exist_ok=True)

            np.save(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease_id, 'deepcombi_auc'), auc_scores)
            np.save(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease_id, 'deepcombi_prec'), prec_scores)
            np.save(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease_id, 'deepcombi_acc'), acc_scores)
            np.save(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease_id, 'deepcombi_balacc'), bal_acc_scores)
            np.save(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease_id, 'combi_auc'), auc_scores_svm)
            np.save(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease_id, 'combi_prec'), prec_scores_svm)
            np.save(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease_id, 'combi_acc'), acc_scores_svm)
            np.save(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease_id, 'combi_balacc'), bal_acc_scores_svm)


    def test_hpsearch_crohn(self, real_genomic_data, real_labels_cat, real_idx):
        """ Runs HP search for a subset of chromosomes
        """
        # python -m pytest -s tests/test_lotr.py::TestLOTR::test_hpsearch_crohn

        disease = 'CD'  # disease_IDs[int(os.environ['SGE_TASK_ID'])-1]

        for chrom in [5]:  # range(1,23):


            data = real_genomic_data(disease, chrom)
            fm = char_matrix_to_featmat(data, '3d', real_pnorm_feature_scaling)
            labels_cat = real_labels_cat(disease)
            idx = real_idx(disease)
            params_space = {
                'n_snps': [fm.shape[1]],
                'epochs': [25, 50, 75, 100],
                'dropout_rate': [0.3],
                'l1_reg': [0.1, 0.01, 0.001],
                'l2_reg': [0],
                'hidden_neurons': [3, 6, 10, 64],
                'lr': [0.00001],
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
                       #reduction_method='gamify',
                       minimize_loss=False,
                       params=params_space,
                       model=talos_wrapper,
                       experiment_name=os.path.join('experiments','MONTAEZ_findCD5','talos',disease,str(chrom)))


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
