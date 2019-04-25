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


class TestCombi(unittest.TestCase):

    _multiprocess_shared_ = True

    data = None
    labels = None
    pvalues = None

    @classmethod
    def setUp(self):
        skiprows = 0
        with open('PythonImplementation/data/data.txt', 'r') as d, open('PythonImplementation/data/labels.txt', 'r') as l:
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

    def test_keras_training(self):
        batch_size = 32
        epochs = 5
        n_individuals, features_dim = self.data.shape

        # Preprocess data
        feature_matrix = string_to_featmat(self.data)
        one_hot_encoded_labels = to_categorical(self.labels)

        model = Sequential()
        model.add(Dense(units=500, activation='relu', input_dim=3*features_dim))
        model.add(Dense(units=2, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])

        dir_path = os.path.dirname(os.path.realpath(__file__))
        tensorboardCb = callbacks.TensorBoard(log_dir=dir_path+'/keras-logs', histogram_freq=0,
                                           write_graph=True, write_images=True)
        earlyStoppingCb = callbacks.EarlyStopping(patience=2, monitor='val_loss')

        history = model.fit(feature_matrix, one_hot_encoded_labels,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=[
                      tensorboardCb, 
                      earlyStoppingCb
                      ],
                  validation_split=0.33)

        model.save(dir_path+'/exported_models/test_model.h5')

    def test_innvestigate_analysis(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        model = load_model(dir_path+'/exported_models/test_model.h5')
        model.get_layer(name='dense_1').name = 'dense_first'
        n_individuals, features_dim = self.data.shape

        # Stripping the softmax activation from the model
        model_wo_sm = iutils.keras.graph.model_wo_softmax(model)

        # Creating an analyzer
        dtaylor_analyzer = innvestigate.analyzer.DeepTaylor(model_wo_sm)
        feature_matrix = string_to_featmat(self.data)

        analysis = dtaylor_analyzer.analyze(feature_matrix)
        x = range(features_dim)
        analysis = analysis.reshape(n_individuals, features_dim, 3).sum(axis=2)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(range(features_dim), analysis[0], marker='x')
        ax1.scatter(range(features_dim), analysis[1], marker='x')
        plt.show()
        print("done")


if __name__ == '__main__':
    unittest.main()
