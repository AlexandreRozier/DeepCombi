

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

classifier = svm.LinearSVC(C=Cs, penalty='l2', tol=svm_epsilon, verbose=0, dual=True)

def svm_step(featmat_2d, labels, filter_window_size, top_k , p):
    """ SVM training + weights postprocessing
    """
    classifier.fit(featmat_2d, labels)
    print("First step: SVM train_acc: {}".format(classifier.score(featmat_2d, labels)))
    weights = classifier.coef_[0] # n_snps * 3
    
    top_indices_sorted, _ = postprocess_weights(weights,top_k, filter_window_size, p_svm, p_pnorm_filter)
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
  


def permuted_combi_method(data, labels, n_permutations, alpha, *args):
    min_pvalues = np.zeros(n_permutations)

    for i in tqdm(range(n_permutations)):
        permuted_labels = random_state.permutation(labels)
        _, pvalues = combi_method(data, permuted_labels, *args)
        min_pvalue = pvalues.min()
        min_pvalues[i] = min_pvalue
    sorted_min_pvalues = np.sort(min_pvalues)

    # Alpha percentile of sorted p-values
    t_star = sorted_min_pvalues[math.ceil(n_permutations*alpha)]
    return t_star
