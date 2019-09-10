

import numpy as np 
from sklearn import svm 
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
import os 
from sklearn.preprocessing import StandardScaler
from parameters_complete import TEST_DIR, svm_epsilon, p_svm, p_pnorm_filter
from helpers import moving_average, chi_square, h5py_to_featmat, postprocess_weights
from parameters_complete import Cs, n_total_snps, random_state
from joblib import Parallel, delayed
import tensorflow
import innvestigate
import innvestigate.utils as iutils
from keras.utils import to_categorical

classifier = svm.LinearSVC(C=Cs, penalty='l2', tol=svm_epsilon, verbose=0, dual=True)

def svm_step(featmat_2d, labels, filter_window_size, top_k , p):
    """ SVM training + weights postprocessing
    """
    classifier.fit(featmat_2d, labels)
    print("First step: SVM train_acc: {}".format(classifier.score(featmat_2d, labels)))
    weights = classifier.coef_[0] # n_snps * 3
    
    top_indices_sorted, _ = postprocess_weights(weights,top_k, filter_window_size, p_svm, p_pnorm_filter)
    return top_indices_sorted


def dnn_step(model,hp, data, featmat_3d, labels, labels_cat, g, filter_window_size, top_k, p):
    
    with tensorflow.Session().as_default():
        # Super time-consuming
        model.fit(...)
        
        model = iutils.keras.graph.model_wo_softmax(model)
        analyzer = innvestigate.analyzer.LRPAlpha1Beta0(model)
        weights = analyzer.analyze(featmat_3d).sum(0)
        
        if np.max(abs(weights)) < 0.005:
            raise Exception("Model failed to train")
        
        top_indices_sorted,_ = postprocess_weights(weights, top_k, filter_window_size, p_svm, p_pnorm_filter)
    
    return top_indices_sorted

def combi_method(data, fm, labels, p, filter_window_size, top_k=0):
    """
    data: (n, n_snps, 2) SNPs-to-person matrix
    labels: (n)
    top_k: Keep K greatest SVM weight vector values
    pnorm_feature_scaling: unused, default to normalizing with norm 2
     
    RETURNS: indices, pvalues 
    """
    print("Performing combi...")
    

    # SVM Step to select the most k promising SNPs
    top_indices_sorted = svm_step(fm, labels, filter_window_size, top_k, p)
    
    # For those SNPs, compute p-values on the second half of the data
    pvalues = chi_square(data[:,top_indices_sorted], labels)

    return top_indices_sorted, pvalues
  
def deepcombi_method(model, data, fm, labels, labels_cat, p, filter_window_size, top_k=0):
  
    top_indices_sorted = dnn_step(model, data, fm, labels, labels_cat, filter_window_size, top_k, p)
    
    pvalues = chi_square(data[:,top_indices_sorted], labels)

    return top_indices_sorted, pvalues



def permuted_combi_method(data, fm, labels, n_permutations, alpha, *args):

    def f():
        permuted_labels = random_state.permutation(labels)
        _, pvalues = combi_method(data, fm, permuted_labels, *args)
        return  pvalues.min()
    
    min_pvalues = Parallel(n_jobs=-1, require='sharedmem')(delayed(f)() for i in tqdm(range(n_permutations)))

    # Alpha percentile of sorted p-values
    t_star = np.quantile(min_pvalues, alpha)

    return t_star

def permuted_deepcombi_method(model, data, fm, labels, labels_cat, n_permutations, alpha, *args):

    def f():
        permuted_labels = random_state.permutation(labels)
        permuted_labels_cat = to_categorical((permuted_labels+1)/2)
        _, pvalues = deepcombi_method(model, data, fm, permuted_labels,permuted_labels_cat, *args)
        return  pvalues.min()
    
    min_pvalues = Parallel(n_jobs=-1, require='sharedmem')(delayed(f)() for i in tqdm(range(n_permutations)))

    # Alpha percentile of sorted p-values
    t_star = np.quantile(min_pvalues, alpha)

    return t_star
