import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import pickle
import h5py
import numpy as np 
import scipy.io
import tensorflow 
from models import create_montaez_dense_model_2, best_params_montaez_2
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from helpers import char_matrix_to_featmat, generate_name_from_params, postprocess_weights
from sklearn.model_selection import ParameterGrid
from parameters_complete import random_state, REAL_DATA_DIR, DATA_DIR, IMG_DIR, nb_of_jobs, pnorm_feature_scaling,p_pnorm_filter, filter_window_size, top_k, p_svm
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
import keras.backend as K
from keras.models import load_model
from Indices import Indices
from combi import  permuted_deepcombi_method, classifier
from joblib import Parallel, delayed
from tqdm import tqdm
from helpers import chi_square, plot_pvalues
import talos 
from talos.model.early_stopper import early_stopper
import innvestigate
import innvestigate.utils as iutils
TEST_PERCENTAGE = 0.20

class TestLOTR(object):


    def test_class_proportions(self, real_labels):
        print("Cases:{}".format(real_labels[real_labels>0.5].sum()))
        print("Controls:{}".format(real_labels[real_labels<0.5].sum()))
        

    def test_train_networks(self, real_h5py_data, real_labels, real_labels_0based, real_labels_cat):
        """ Runs HP search for a subset of chromosomes (on CPU, high degree of paralellism.)
        """
        # Each node gets a set of chromosomes to process :D

        chrom=int(os.environ['SGE_TASK_ID'])
        print('loading..')
        
        # 1. Do hyperparam search on each chromosome and find parameters with BEST VAL ACCURAC
                                                            
        data = real_h5py_data(chrom)                                                                                                                                                                                                                                                                                                                                                          
        fm = char_matrix_to_featmat(data, '3d')
        n_subjects = fm.shape[0]

        
        print('loaded')

        splitter =  StratifiedShuffleSplit(n_splits=1, test_size = TEST_PERCENTAGE, random_state=random_state)

        train_indices, test_indices = next(splitter.split(np.zeros(n_subjects), real_labels_0based))
        idx = Indices(train_indices, test_indices, None)                                                                             
        
        params_space = {
            'n_snps':[fm.shape[1]],
            'epochs': [1000],
            #'patience': [30,50,100],
            #'factor':np.linspace(0.1,0.8,7),
            'dropout_rate':[0.3],
            'l1_reg': list(np.logspace(-6, -2, 5)),
            'l2_reg': list(np.logspace(-6, -2, 5)),
            'lr' : list(np.logspace(-4, -2, 3)),        
        }
        
        
        def talos_wrapper(x,y,x_val,y_val,params):
            model = create_montaez_dense_model_2(params)
            out = model.fit(x=x,
                    y=y,
                    validation_data=(x_val, y_val),
                    epochs=params['epochs'],
                    #callbacks=[early_stopper(params['epochs'])]
            )
            return out, model



        talos.Scan(x=fm[idx.train],
                y=real_labels_cat[idx.train],
                x_val=fm[idx.test], 
                y_val=real_labels_cat[idx.test],
                params=params_space,
                model=talos_wrapper,
                experiment_name='no_es/chrom_{}'.format(chrom))
        

    def test_parameters(self, real_h5py_data, real_labels_cat, real_idx):

        data = real_h5py_data(1)                                                                                                                                                                                                                                                                                                                                                          
        fm = char_matrix_to_featmat(data, '3d')
        best_params_montaez_2['n_snps'] = fm.shape[1]
        model = create_montaez_dense_model_2(best_params_montaez_2)

        model.fit(x=fm[real_idx.train],
                y=real_labels_cat[real_idx.train],
                validation_data=(fm[real_idx.test], real_labels_cat[real_idx.test]),

                epochs=best_params_montaez_2['epochs'],
                callbacks=[
                    TensorBoard(log_dir=os.path.join(REAL_DATA_DIR,'tb','chrom1-useless'),
                        histogram_freq=3,
                        write_graph=False,
                        write_grads=True,
                        write_images=False),
                    ReduceLROnPlateau(monitor='val_loss', 
                        factor=best_params_montaez_2['factor'], 
                        patience=best_params_montaez_2['patience'], 
                        mode='min'
                    ),
                ],
                verbose=1)

    def test_permutations(self, real_h5py_data, real_labels, real_labels_0based, real_labels_cat, real_idx, alphas, alphas_EV):
        """ Computes t_star for each chromosome thanks to the permutation method.
        """

        chrom = int(os.environ['SGE_TASK_ID'])

        # 1. Do hyperparam search on each chromosome and find parameters with BEST VAL ACCURAC

        data = real_h5py_data(chrom)                                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                    
        fm = char_matrix_to_featmat(data, '3d')

        n_permutations = 3
        alpha_sig = float(alphas[chrom])
        alpha_sig_EV = float(alphas_EV[chrom])
        #hp = pickle.load(open(os.path.join(REAL_DATA_DIR,'hyperparams','chrom{}.p'.format(chrom)),'rb'))
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

        t_star = permuted_deepcombi_method(model, hp, real_h5py_data, fm, real_labels, real_labels_cat, n_permutations, alpha_sig, pnorm_feature_scaling, filter_window_size, top_k, mode='min')
        t_star_EV = permuted_deepcombi_method(model,hp, real_h5py_data, fm, real_labels, real_labels_cat, n_permutations, alpha_sig_EV, pnorm_feature_scaling, filter_window_size, top_k, mode='all')
        pickle.dump(t_star, open(os.path.join(REAL_DATA_DIR,'chrom{}-t_star.p'.format(chrom)),'wb'))
        pickle.dump(t_star_EV, open(os.path.join(REAL_DATA_DIR,'chrom{}-t_star_EV.p'.format(chrom)),'wb'))


    def test_generate_plots(self):
        diseases_id =['BD'] # ['BD', 'CAD','CD','HT','RA','T1D','T2D']
        diseases = ['Bipolar disorder'] #['Bipolar disorder', 'Coronary artery disease','Crohns disease','Hypertension','Rheumatoid arthritis','Type 1 Diabetes','Type 2 diabetes']
        chromosomes_to_plot = [22] # range(1,23)

        # for disease in diseases_id:
        disease = ['BD']
        #TODO report chunk vs non chunk
        

        chrom_fig = plt.figure()
        raw_pvalues_ax = chrom_fig.add_subplot(131)
        pp_pvalues_ax = chrom_fig.add_subplot(132)
        pp_rm_ax = chrom_fig.add_subplot(133)

        offset = 0
        middle_offset_history = np.zeros(22)

        for chromo in range(1,23):
            #t_star = pickle.load(open(os.path.join(REAL_DATA_DIR,disease,'chrom{}-t_star.p'.format(chrom)),'rb'))
            #t_star_EV = pickle.load(open(os.path.join(REAL_DATA_DIR,disease,'chrom{}-t_star_EV.p'.format(chrom)),'rb'))
            #raw_rm = np.load(open(os.path.join(REAL_DATA_DIR, disease, 'chrom{}-rm.p'.format(chrom)),'rb'))

            
            raw_rm = np.random.rand(100,3)
            raw_pvalues = np.random.rand(100)

            n_snps = raw_rm.shape[0]
            
            top_indices, pp_rm = postprocess_weights(raw_rm,top_k, filter_window_size, p_svm, p_pnorm_filter) # TODO check those values
            #raw_pvalues = np.load(open(os.path.join(REAL_DATA_DIR, disease, 'chrom{}-rpv.p'.format(chrom)),'rb'))

            #if disease=='CAD' and chromo!=9:
            #    raw_pvalues[-log10(raw_pvalues) > 6] = 1

            postprocessed_pvalues = np.ones(n_snps)
            postprocessed_pvalues[top_indices] = raw_pvalues[top_indices] # HERE pp_pvalues are obtained by filtering already computed pvalues. in combi its different, we chi_square only on the preselected pvalues

            color = 'b' if (chromo % 2 ==0) else 'r'
            raw_pvalues_ax.scatter(range(offset, offset+n_snps), -np.log10(raw_pvalues), c=color)
            pp_pvalues_ax.scatter(range(offset, offset+n_snps), -np.log10(postprocessed_pvalues), c=color)
            pp_rm_ax.scatter(range(offset, offset+n_snps), pp_rm, c=color)
            middle_offset_history[chromo-1] = offset +  int(n_snps/2)

            offset += n_snps
        
        plt.setp(raw_pvalues_ax, xticks=middle_offset_history, xticklabels=range(1,23))
        plt.setp(pp_pvalues_ax, xticks=middle_offset_history, xticklabels=range(1,23))
        plt.setp(pp_rm_ax, xticks=middle_offset_history, xticklabels=range(1,23))
        raw_pvalues_ax.set_xlabel('Chromosome')
        raw_pvalues_ax.set_title('Raw p-value thresholding')
        pp_pvalues_ax.set_xlabel('Chromosome')
        pp_pvalues_ax.set_title('DeepCOMBI method')

        pp_rm_ax.set_xlabel('Chromosome')
        pp_rm_ax.set_title('DeepCOMBI method')

        chrom_fig.savefig(os.path.join(REAL_DATA_DIR,'plots', 'manhattan.png'))


    def test_svm_accuracy(self, real_h5py_data, real_labels, real_labels_0based, real_idx):
        data = real_h5py_data(1)                                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                        
        x = char_matrix_to_featmat(data, '2d')
        print('Fitting data...')
        svm_model = classifier.fit(x[real_idx.train], real_labels[real_idx.train])
        print(svm_model.score(x[real_idx.test], real_labels[real_idx.test]))


    def test_plot_all_pvalues(self, real_h5py_data, real_labels, alphas):
        """ Compares efficiency of the combi method with several TTBR
        """
        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches(18.5, 10.5)
        axes.axhline(y=5)
        idx=0
        for i in range(1,5):       
            h5py_data = real_h5py_data(i)
            complete_pvalues = chi_square(h5py_data, real_labels)
            informative_idx = np.argwhere(complete_pvalues < 1e-5)
            color = np.zeros((len(complete_pvalues),3))
            color[:] = [255,0,0] if (i % 2 ==0) else [0,0,255]
            color[informative_idx] = [0,255,0]
            
            axes.scatter(range(idx, idx+len(complete_pvalues)), -np.log10(complete_pvalues), c=color/255.0, marker='x')
            idx += len(complete_pvalues)
        fig.savefig(os.path.join(IMG_DIR, 'genome-wide-pvalues.png'))


    def test_lrp(self, real_h5py_data, real_labels,real_labels_0based,real_labels_cat, real_idx, rep, tmp_path):
        """ Compares efficiency of the combi method with several TTBR
        """
        fig, axes = plt.subplots(4, 1, sharex='col')
        
        h5py_data = real_h5py_data(1)                                                                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                                                       
        x_2d = char_matrix_to_featmat(h5py_data, '2d')
        x_3d = char_matrix_to_featmat(h5py_data, '3d')
        
        
        g = best_params_montaez_2
        g['l1_reg']=1e-5
        g['l2_reg']=1e-4
        g['lr']=0.0001
        g['epochs']=108
        g['n_snps']=x_3d.shape[1]
        

        model = create_montaez_dense_model_2(g)
        model.fit(x=x_3d[real_idx.train],
                y=real_labels_cat[real_idx.train],
                validation_data=(x_3d[real_idx.test], real_labels_cat[real_idx.test]),
                epochs=g['epochs']
        )
        
        model = iutils.keras.graph.model_wo_softmax(model)
        analyzer = innvestigate.analyzer.LRPAlpha1Beta0(model)
        weights = analyzer.analyze(x_3d).sum(0)

        top_indices_sorted, postprocessed_weights = postprocess_weights(
            weights, top_k, filter_window_size, p_svm, p_pnorm_filter)

        complete_pvalues = chi_square(h5py_data, real_labels)
        axes[0].plot(-np.log10(complete_pvalues))
        axes[0].set_title('RPV')

        # Plot distribution of relevance
        axes[1].plot(np.absolute(weights).reshape(-1,3).sum(1))
        axes[1].set_title('Absolute relevance')

        # Plot distribution of postprocessed vectors
        axes[2].plot(postprocessed_weights)
        axes[2].set_title('Postprocessed relevance')


        # Plot distribution of svm weights
        svm_weights = classifier.fit(x_2d, real_labels).coef_
        axes[3].plot(np.absolute(svm_weights).reshape(-1,3).sum(1))
        axes[3].set_title('Absolute SVM weight')

    
        fig.savefig(os.path.join(IMG_DIR, 'manhattan-real.png'))