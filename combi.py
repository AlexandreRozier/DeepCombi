

from string_to_featmat import string_to_featmat
import numpy as np 
from sklearn import preprocessing as pp, svm 
from helpers import moving_average
from chi_square_goodness_of_fit_test import chi_square_goodness_of_fit_test
from matplotlib import pyplot as plt

def combi_method(data, labels,top_k, pnorm_feature_scaling, svm_rep, Cs, p, classy, filter_window_size,p_pnorm_filter ):
    """
    data: np.array(n,3*p) SNPs-to-person matrix
    labels: np.array(n)
    top_k: Keep K greatest SVM weight vector values
    pnorm_feature_scaling: unused, default to normalizing with norm 2
    svm_rep: 
    """
    
    (num_subj,num_snps) = data.shape


    # Run Combi-Method and identify top_k best SNPs
    ### string data to feature_matrix ###
    featmat = string_to_featmat( data )
    featmat = featmat.astype(np.double)
    featmat = pp.scale(featmat, axis=0) # preprocess matrix   

    ### SVM training ###
    classifier = svm.LinearSVC()
    classifier.fit(featmat, labels)

    ### Postprocessing with Moving Average Filter ###
    weights = classifier.coef_[0]
    weights = np.mean(weights.reshape(-1, 3), axis=1) # Group & Average weights by 3 (yields locus's importance measure)
    weights = np.abs(weights)
    weights = moving_average(weights,filter_window_size, p) 
    top_index_sorted = weights.argsort()[::-1][:top_k] # Gets indices of top_k greatest elements
    assert(weights[top_index_sorted[0]] == np.amax(weights))
    

    # 3. For those SNPs compute p-values on the second half of the data
    # TODO parallelize 
    pvalues_all = chi_square_goodness_of_fit_test(data, labels)
    color_array = ['b' if i in top_index_sorted else 'r' for i, pvalue in enumerate(pvalues_all) ]
    plt.scatter(np.linspace(0,1,len(pvalues_all)),-np.log10(pvalues_all), color=color_array)

    plt.show()
    print("hallo")

    ### Compute p-values Bonferoni ###
    """
    pvalues_final_filter = process_pvalues(pvalues_all,selector_filter)
    pvalues_final = process_pvalues(pvalues_all,selector)
    """