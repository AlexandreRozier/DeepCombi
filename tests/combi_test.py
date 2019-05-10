import unittest
import numpy as np
from parameters_complete import pnorm_feature_scaling, svm_rep, Cs, classy, filter_window_size, p_pnorm_filter
from helpers import chi_square, string_to_featmat
from combi import combi_method, permuted_combi_method
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import callbacks
from keras.models import load_model
import math
import os
import innvestigate
import innvestigate.utils as iutils
import matplotlib.pyplot as plt
import pytest

@pytest.mark.incremental
class TestCombi(object):

    
   
    def test_pvalues_generation(self, raw_data, raw_labels):
        pvalues = chi_square(raw_data, raw_labels)
        assert (-np.log10(pvalues)).max() > 20

    def test_pvalues_subset_generation(self, raw_data, raw_labels):
        n_pvalues = 30
        indices = np.random.randint(raw_data.shape[1], size=n_pvalues)
        pvalues = chi_square(raw_data, raw_labels, indices)

    def test_combi(self,raw_data, raw_labels):
        n_pvalues = 30
        top_indices_sorted, pvalues = combi_method(raw_data, raw_labels, pnorm_feature_scaling, svm_rep,
                               Cs, 2, classy, filter_window_size, p_pnorm_filter, n_pvalues,full_plot=True)
        print('PVALUES CHOSEN: {}'.format(top_indices_sorted))


    def test_permutations(self, raw_data, raw_labels):
        """ Tests the permuted version of COMBI
        """
        n_permutations = 5
        alpha = 0.05
        n_pvalues = 30
        t_star = permuted_combi_method(raw_data, raw_labels,
                                n_permutations, alpha, n_pvalues,pnorm_feature_scaling, svm_rep,
                               Cs, 2, classy, filter_window_size, p_pnorm_filter, n_pvalues)
        pvalues = chi_square(raw_data, raw_labels)
        plt.scatter(range(len(pvalues)),-np.log10(pvalues), marker='x')
        plt.axhline(y=-np.log10(t_star), color='r', linestyle='-')
        plt.show()
        assert t_star > 0
        assert t_star < pvalues.mean() # Tests t_star selectivity

