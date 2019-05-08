

import numpy as np 
from sklearn import preprocessing as pp, svm 
from helpers import moving_average, chi_square, string_to_featmat
from matplotlib import pyplot as plt


def compute_top_k_indices(data, labels, filter_window_size, top_k , p):
    # Run Combi-Method and identify top_k best SNPs
    ### string data to feature_matrix ###
    featmat = string_to_featmat( data )
      
    ### SVM training ###
    classifier = svm.LinearSVC()
    classifier.fit(featmat, labels)

    ### Postprocessing with Moving Average Filter ###
    weights = classifier.coef_[0]
    weights = np.mean(weights.reshape(-1, 3), axis=1) # Group & Average weights by 3 (yields locus's importance measure)
    weights = np.abs(weights)
    weights = moving_average(weights,filter_window_size, p) 
    top_indices_sorted = weights.argsort()[::-1][:top_k] # Gets indices of top_k greatest elements
    assert(weights[top_indices_sorted[0]] == np.amax(weights))
    return top_indices_sorted

def combi_method(data, labels, pnorm_feature_scaling, svm_rep, Cs, p, classy, filter_window_size,p_pnorm_filter, top_k=0, full_plot=False ):
    """
    data: np.array(n,3*p) SNPs-to-person matrix
    labels: np.array(n)
    top_k: Keep K greatest SVM weight vector values
    pnorm_feature_scaling: unused, default to normalizing with norm 2
     
    RETURNS: indices, pvalues 
    """
    
    
    # Avoid SVM preprocessing
    if(top_k==0):
        return chi_square(data, labels)

    top_indices_sorted = compute_top_k_indices(data, labels, filter_window_size, top_k, p)
    # 3. For those SNPs compute p-values on the second half of the data

    pvalues = chi_square(data, labels, top_indices_sorted)

    if full_plot:
        complete_pvalues = chi_square(data, labels)
        color_array = ['b' if i in top_indices_sorted else 'r' for i, pvalue in enumerate(complete_pvalues) ]
        plt.scatter(range(len(complete_pvalues)),-np.log10(complete_pvalues), marker='x',color=color_array)
        plt.xlim(4800,5200)
        plt.show()

    return top_indices_sorted, pvalues
  