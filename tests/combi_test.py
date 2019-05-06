import unittest
import numpy as np
from parameters_complete import pnorm_feature_scaling, svm_rep, Cs, classy, filter_window_size, p_pnorm_filter
from helpers import chi_square, permuted_combi, string_to_featmat
from combi import combi_method
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
from hyperas.distributions import uniform
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim


class TestCombi(unittest.TestCase):

    _multiprocess_shared_ = True

    data = None
    labels = None
    pvalues = None

    @classmethod
    def setUp(self):
        skiprows = 0
        with open('./data/data.txt', 'r') as d, open('./data/labels.txt', 'r') as l:
            self.data = np.loadtxt(d, np.chararray, skiprows=skiprows)
            self.labels = np.loadtxt(l, dtype=np.int8, skiprows=skiprows)

    def test_pvalues_generation(self):
        pvalues = chi_square(self.data, self.labels)
        self.assertGreater((-np.log10(pvalues)).max(), 20)

    def test_pvalues_subset_generation(self):
        n_pvalues = 30
        indices = np.random.randint(self.data.shape[1], size=n_pvalues)
        pvalues = chi_square(self.data, self.labels, indices)

    def test_combi(self):
        n_pvalues = 30
        pvalues = combi_method(self.data, self.labels, pnorm_feature_scaling, svm_rep,
                               Cs, 2, classy, filter_window_size, p_pnorm_filter, n_pvalues)

    def test_permutations(self):
        n_permutations = 10
        alpha = 0.05
        n_pvalues = 30
        t_star = permuted_combi(self.data, self.labels,
                                n_permutations, alpha, n_pvalues)
        self.assertLess(t_star, 1)
        self.assertGreater(t_star, 0)

