

import numpy as np 
from sklearn import preprocessing as pp, svm 
from helpers import moving_average, chi_square, string_to_featmat
from parameters_complete import TEST_DIR
import matplotlib
matplotlib.use('QT5Agg') 
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
import os 
from parameters_complete import Cs, n_total_snps
from sklearn.preprocessing import StandardScaler

def compute_top_k_indices(data, labels, filter_window_size, top_k , p):
    # Run Combi-Method and identify top_k best SNPs
    ### string data to feature_matrix ###
    featmat = string_to_featmat( data )
    
    ### SVM training ###
    classifier = svm.LinearSVC(C=Cs, penalty='l2', verbose=0, dual=True)
    classifier.fit(featmat, labels)

    ### Postprocessing with Moving Average Filter ###
    weights = classifier.coef_[0] # n_snps * 3
    weights = np.mean(weights.reshape(-1, 3), axis=1) # Group & Average weights by 3 (yields locus's importance measure)
    weights = np.abs(weights)
    weights = moving_average(weights,filter_window_size) 
    top_indices_sorted = weights.argsort()[::-1][:top_k] # Gets indices of top_k greatest elements
    assert(len(weights) == n_total_snps)
    assert(weights[top_indices_sorted[0]] == np.amax(weights))
    return top_indices_sorted

def combi_method(data, labels, pnorm_feature_scaling, p, classy, filter_window_size,p_pnorm_filter, top_k=0, full_plot=False ):
    """
    data: (n, n_snps, 2) SNPs-to-person matrix
    labels: (n)
    top_k: Keep K greatest SVM weight vector values
    pnorm_feature_scaling: unused, default to normalizing with norm 2
     
    RETURNS: indices, pvalues 
    """
    
    # Avoid SVM preprocessing
    if(top_k==0):
        return chi_square(data, labels)
    
    # SVM Step to select the most k promising SNPs
    print("Performing SVM...")
    top_indices_sorted = compute_top_k_indices(data, labels, filter_window_size, top_k, p)
    
    # For those SNPs, compute p-values on the second half of the data
    print("Performing partial X2...")
    pvalues = chi_square(data[:,top_indices_sorted], labels)

    if full_plot:
        print("Performing complete X2 to prepare plotting...")

        complete_pvalues = chi_square(data, labels)
        color_array = ['b' if i in top_indices_sorted else 'r' for i, pvalue in enumerate(complete_pvalues) ]
        plt.scatter(range(len(complete_pvalues)),-np.log10(complete_pvalues), marker='x',color=color_array)
        #plt.xlim(4800,5200)
        plt.savefig(os.path.join(TEST_DIR,'combi.png'))

    return top_indices_sorted, pvalues
  

def permuted_combi_method(data, labels, n_permutations, alpha, n_pvalues, *args):
    min_pvalues = np.zeros(n_permutations)
    for i in tqdm(range(n_permutations)):
        permuted_labels = np.random.permutation(labels)
        indices, pvalues = combi_method(data, permuted_labels, *args)
        min_pvalue = pvalues.min()
        min_pvalues[i] = min_pvalue
    sorted_min_pvalues = np.sort(min_pvalues)

    # Alpha percentile of sorted p-values
    t_star = sorted_min_pvalues[math.ceil(n_permutations*alpha)]
    return t_star