

import numpy as np 
from sklearn import svm 
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
import os 
from sklearn.preprocessing import StandardScaler
from helpers import moving_average, chi_square, h5py_to_featmat, postprocess_weights, po
from parameters_complete import TEST_DIR, svm_epsilon, Cs, real_Cs, n_total_snps, random_state
from joblib import Parallel, delayed
import tensorflow
import innvestigate
import innvestigate.utils as iutils
from keras.utils import to_categorical

toy_classifier = svm.LinearSVC(C=Cs, penalty='l2', tol=svm_epsilon, verbose=0, dual=True)
real_classifier = svm.LinearSVC(C=real_Cs, penalty='l2', tol=svm_epsilon, verbose=0, dual=True)



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
    
    top_indices_sorted, _ = postprocess_weights_without(raw_weights,top_k, filter_window_size, psvm, pnorm_filter)
    
    # For those SNPs, compute p-values on the second half of the data
    pvalues = chi_square(data[:,top_indices_sorted], labels)

    return top_indices_sorted, pvalues, raw_weights.reshape(-1, 3)


def deepcombi_method(model, data, fm, labels, filter_window_size, pnorm_filter, psvm, top_k):
        
    for i, layer in enumerate(model.layers):
            if layer.name == 'dense_1':
                layer.name = 'blu{}'.format(str(i))
            if layer.name == 'dense_2':
                layer.name = 'bla{}'.format(str(i))
            if layer.name == 'dense_3':
                layer.name = 'bleurg{}'.format(str(i))

    model = iutils.keras.graph.model_wo_softmax(model)
    analyzer = innvestigate.analyzer.LRPAlpha1Beta0(model)
    raw_weights = analyzer.analyze(fm).sum(0)
    
    top_indices_sorted,_ = postprocess_weights(raw_weights, top_k, filter_window_size, psvm, pnorm_filter)
    
    pvalues = chi_square(data[:,top_indices_sorted], labels)

    return top_indices_sorted, pvalues, raw_weights



def permuted_combi_method(data, fm, labels, filter_window_size, pnorm_filter, psvm, top_k, n_permutations, alpha):

    def f():
        permuted_labels = random_state.permutation(labels)
        _, pvalues,_ = combi_method(classifier, data,  fm, permuted_labels, filter_window_size, pnorm_filter, psvm, top_k)
        return  pvalues.min()
    
    min_pvalues = Parallel(n_jobs=-1, require='sharedmem')(delayed(f)() for i in tqdm(range(n_permutations)))

    # Alpha percentile of sorted p-values
    t_star = np.quantile(min_pvalues, alpha)

    return t_star


def permuted_deepcombi_method(model, data, fm, labels, labels_cat, filter_window_size, pnorm_filter, psvm, top_k, n_permutations, alpha, mode='min'):

    def f():
        permuted_labels = random_state.permutation(labels)
        permuted_labels_cat = to_categorical((permuted_labels+1)/2)
        _, pvalues, _ = deepcombi_method(model, data, fm, permuted_labels, filter_window_size, pnorm_filter, psvm, top_k)
        if mode=='min':
            return  np.array(pvalues.min())
        elif mode=='all':
            return np.array(pvalues)
    
    min_pvalues = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(f)() for i in tqdm(range(n_permutations)))).flatten()

    # Alpha percentile of sorted p-values
    t_star = np.quantile(min_pvalues, alpha)

    return t_star
