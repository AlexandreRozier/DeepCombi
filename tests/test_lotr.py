import os
import pickle
import h5py
import numpy as np 
import scipy.io
import tensorflow 

from models import create_montaez_dense_model_2
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from helpers import char_matrix_to_featmat, generate_name_from_params, postprocess_weights
from sklearn.model_selection import ParameterGrid
from parameters_complete import random_state, REAL_DATA_DIR, DATA_DIR, nb_of_jobs, pnorm_feature_scaling,p_pnorm_filter, filter_window_size, top_k, p_svm
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
import keras.backend as K
from keras.models import load_model
from Indices import Indices
from combi import  permuted_deepcombi_method
from joblib import Parallel, delayed
from tqdm import tqdm
from matplotlib import pyplot as plt

TEST_PERCENTAGE = 0.20

class TestLOTR(object):


    def test_train_networks(self, real_h5py_data, real_labels, real_labels_0based, real_labels_cat):
        """ Runs HP search for a subset of chromosomes (on CPU, high degree of paralellism.)
        """
        # Each node gets a set of chromosomes to process :D
        chroms_per_node = np.array_split(range(1,23), nb_of_jobs)

        for chrom in chroms_per_node[int(os.environ['SGE_TASK_ID'])-1]:
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
                'patience': [10,30,50,100],
                'factor':np.linspace(0.1,0.8,9),
                'dropout_rate':np.linspace(0.1,0.5,5),
                'l1_reg': np.logspace(-6, -2, 5),
                'l2_reg': np.logspace(-6, -2, 5),
                'lr' : np.logspace(-4, -2, 3),        
            }

            grid = ParameterGrid(params_space)
            BUDGET = 100
            print("Testing {} % of the hp space".format(BUDGET*100/len(grid)))
            grid = random_state.choice(list(grid), BUDGET)

            
            def train(g):
                with tensorflow.Session().as_default():
                    with tensorflow.Session().graph.as_default():
                        model = create_montaez_dense_model_2(g)

                        history = model.fit(x=fm[idx.train],
                                y=real_labels_cat[idx.train],
                                validation_data=(fm[idx.test], real_labels_cat[idx.test]),
                
                                epochs=g['epochs'],
                                callbacks=[
                                    TensorBoard(log_dir=os.path.join(REAL_DATA_DIR,'tb','chrom{}-{}'.format(chrom,generate_name_from_params(g))),
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
                                verbose=1)
                        return history.history['val_acc'][-1]

                    
            accuracies = Parallel(n_jobs=30)(delayed(train)(g) for g in tqdm(grid))
            best_g = grid[np.argmax(accuracies)]
            pickle.dump(best_g, open(os.path.join(REAL_DATA_DIR,'hyperparams','chrom{}.p'.format(chrom)),'wb'))
            


    def test_permutations(self, real_h5py_data, real_labels, real_labels_0based, real_labels_cat, real_idx):
        """ Computes t_star for each chromosome thanks to the permutation method.
        """

        chroms_per_node = np.array_split(range(1,23), nb_of_jobs)
        alphas = scipy.io.loadmat(os.path.join(DATA_DIR,'alpha_j.mat'))['alpha_j'].T[0]
        alphas_EV = scipy.io.loadmat(os.path.join(DATA_DIR,'alpha_j_EV.mat'))['alpha_j_EV'].T[0]

        for chrom in chroms_per_node[int(os.environ['SGE_TASK_ID'])-1]:

            # 1. Do hyperparam search on each chromosome and find parameters with BEST VAL ACCURAC

            data = real_h5py_data(chrom)                                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                        
            fm = char_matrix_to_featmat(data, '3d')

            n_permutations = 1000
            alpha_sig = float(alphas[chrom])
            alpha_sig_EV = float(alphas_EV[chrom])
            hp = pickle.load(open(os.path.join(REAL_DATA_DIR,'hyperparams','chrom{}.p'.format(chrom)),'rb'))
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

        # Load per-chromosome raw relevance mappings
        # If cad remove stuff
        # Build postprocessed r.m.
        # Filter out snps thanks to permuted threshold

        # plot per chromosome rpv
        # plot per chromosome combi pp pv
        # plot per chromosome combi pp pv


