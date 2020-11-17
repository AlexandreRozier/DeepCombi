

import matplotlib
import numpy as np
from sklearn import svm
import pdb

matplotlib.use('Agg')
from tqdm import tqdm

from helpers import char_matrix_to_featmat, chi_square, postprocess_weights
from parameters_complete import FINAL_RESULTS_DIR, svm_epsilon, Cs, real_Cs, random_state
from models import create_montaez_dense_model
from sklearn.utils import class_weight
import os
from keras.callbacks import CSVLogger

from math import ceil

from joblib import Parallel, delayed
import innvestigate
import innvestigate.utils as iutils
from keras.utils import to_categorical

toy_classifier = svm.LinearSVC(C=Cs, penalty='l2', loss='hinge', tol= svm_epsilon,dual=True, verbose=0)
real_classifier = svm.LinearSVC(C=real_Cs,penalty='l2', loss='hinge', tol=svm_epsilon,dual=True, verbose=0)



def combi_method(classifier,data, fm, labels, filter_window_size, pnorm_filter, psvm, top_k):
    """
    data: (n, n_snps, 2) SNPs-to-person matrix
    labels: (n)
    top_k: Keep K greatest SVM weight vector values
     
    RETURNS: indices, pvalues 
    """
    # SVM Step to select the most k promising SNPs
    classifier.fit(fm, labels)
    raw_weights = classifier.coef_[0] # n_snps * 3
    
    selected_indices_sorted, _ = postprocess_weights(raw_weights,top_k, filter_window_size, psvm, pnorm_filter)
    
    # For those SNPs, compute p-values on the second half of the data
    pvalues = chi_square(data[:,selected_indices_sorted], labels)

    return selected_indices_sorted, pvalues, raw_weights.reshape(-1, 3)


def deepcombi_method(model, data, fm, labels, filter_window_size, pnorm_filter, psvm, top_k):
        
    for i, layer in enumerate(model.layers):
            if layer.name == 'dense_1':
                layer.name = 'blu{}'.format(str(i))
            if layer.name == 'dense_2':
                layer.name = 'bla{}'.format(str(i))
            if layer.name == 'dense_3':
                layer.name = 'bleurg{}'.format(str(i))
    #pdb.set_trace()
    model = iutils.keras.graph.model_wo_softmax(model)
    analyzer = innvestigate.analyzer.LRPAlpha2Beta1(model)
    raw_weights = np.absolute(analyzer.analyze(fm)).sum(0) # n_snps * 3
    
    selected_indices_sorted,_ = postprocess_weights(raw_weights, top_k, filter_window_size, psvm, pnorm_filter)
    
    pvalues = chi_square(data[:,selected_indices_sorted], labels)

    return selected_indices_sorted, pvalues, raw_weights



def permuted_combi_method(data, labels, real_pnorm_feature_scaling, filter_window_size, pnorm_filter, psvm, top_k, n_permutations, alpha, mode='min'):

    def f():
        permuted_labels = random_state.permutation(labels)
        fm_2D = char_matrix_to_featmat(data, '2d', real_pnorm_feature_scaling)

        _, pvalues,_ = combi_method(real_classifier, data,  fm_2D, permuted_labels, filter_window_size, pnorm_filter, psvm, top_k)

        if mode=='min':
            return  np.array(pvalues.min())
        elif mode=='all':
            return np.array(pvalues)
        
    min_pvalues = Parallel(n_jobs=1, require='sharedmem')(delayed(f)() for i in tqdm(range(n_permutations)))
	
    # Alpha percentile of sorted p-values
    min_pvalues_sorted =  np.sort(min_pvalues)
    t_star = min_pvalues_sorted[max(ceil(n_permutations*alpha)-1,1)]
    #pdb.set_trace()

    #t_star = np.nanquantile(min_pvalues, alpha)

    return t_star


def permuted_deepcombi_method(data, labels, disease_id, chrom, pvalue_threshold,real_pnorm_feature_scaling, filter_window_size, pnorm_filter, psvm, top_k, n_permutations, alpha, mode='min'):

    def f():
        permuted_labels = random_state.permutation(labels)
        permuted_labels_cat = to_categorical((permuted_labels+1)/2)
        # pvalues
        rpvt_pvalues = chi_square(data, permuted_labels)

        # pvalue thresholding
        valid_snps = rpvt_pvalues < pvalue_threshold
        data_now = data[:,valid_snps,:]
		
		# Centering and Scaling
        fm = char_matrix_to_featmat(data_now, '3d', real_pnorm_feature_scaling)
        fm[fm>0]=1
        fm[fm<0]=0.
        mean = np.mean(np.mean(fm,0),0)
        std = np.std(np.std(fm,0),0)
        fm = (fm-mean)/std
		
        hp =  {'dropout_rate': 0.3, 'epochs': 500, 'hidden_neurons': 64, 'l1_reg': 0.001, 'l2_reg': 0.0001, 'lr': 1e-05, 'n_snps': int(fm.shape[1])} 
		
        # Optimal parameters for pthresh = 1e-02 
        # hp =  {'dropout_rate': 0.3, 'epochs': 500, 'hidden_neurons': 64, 'l1_reg': 0.001, 'l2_reg': 0.0001, 'lr': 1e-05, 'n_snps': int(fm.shape[1])} 
		
        # Create Model
        model = create_montaez_dense_model(hp)

        # Class weights
        y_integers = np.argmax(permuted_labels_cat, axis=1)
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
        d_class_weights = dict(enumerate(class_weights))

        # Train
        model.fit(x=fm, y=permuted_labels_cat, epochs=hp['epochs'], verbose=0, class_weight=d_class_weights)		
		
        _, pvalues, _ = deepcombi_method(model, data_now, fm, permuted_labels, filter_window_size, pnorm_filter, psvm, top_k)
        if mode=='min':
            return  np.array(pvalues.min())
        elif mode=='all':
            return np.array(pvalues)
    
	#, require='sharedmem'
    min_pvalues = np.array(Parallel(n_jobs=1)(delayed(f)() for i in tqdm(range(n_permutations)))).flatten()

    # Alpha percentile of sorted p-values
    min_pvalues_sorted =  np.sort(min_pvalues)
    t_star = min_pvalues_sorted[max(ceil(n_permutations*alpha)-1,1)]

    #t_star = np.nanquantile(min_pvalues, alpha)

    return t_star
