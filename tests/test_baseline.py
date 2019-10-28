import os

import matplotlib
import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from combi import toy_classifier
from parameters_complete import (Cs, PARAMETERS_DIR, seed)

matplotlib.use('Agg')


class TestBaselines(object):
    """
    This class tests the validation accuracy of several baselines on the synthetic dataset.
    SVM, kernel SVM, Linear Regression,  Decision Trees, Dense Neural Network
    """

    def test_svm_baseline(self, syn_fm, syn_labels, indices):
        """
        SVM needs 2D data and does not support one hot encoded labels
        """
        features = syn_fm('2d')['0'][:]
        labels = syn_labels['0'][:]
        idx = indices['0']

        c_candidates = np.logspace(-5, -1, num=20)

        # More features than datapoints
        val_scores = np.zeros(len(c_candidates))
        train_scores = np.zeros(len(c_candidates))
        for i, c in tqdm(enumerate(c_candidates)):
            toy_classifier.fit(features[idx.train], labels[idx.train])
            train_scores[i] = toy_classifier.score(features[idx.train], labels[idx.train])
            val_scores[i] = toy_classifier.score(features[idx.test], labels[idx.test])

        max_idx = np.argmax(val_scores)
        print("SVM train_acc: {} (std={}) ; obtained with C={} ".format(train_scores[max_idx], np.std(train_scores),
                                                                        train_scores[max_idx]))
        print("SVM val_acc: {} (std={}) ; obtained with C={}".format(val_scores[max_idx], np.std(val_scores),
                                                                     c_candidates[max_idx]))

    def test_kernel_svm_baseline(self, syn_fm, syn_labels, indices):
        """
        SVM needs 2D data and does not support one hot encoded labels
        """
        features = syn_fm('2d')['0'][:]
        labels = syn_labels['0'][:]
        idx = indices['0']

        c_candidates = np.logspace(-5, 0, 10)

        scores = np.zeros(len(c_candidates))
        for i, c in tqdm(enumerate(c_candidates)):
            toy_classifier.fit(features[idx.train], labels[idx.train])
            scores[i] = toy_classifier.score(features[idx.test], labels[idx.test])

        max_idx = np.argmax(scores)

        print("Kernel SVM val_acc: {} obtained with C={}".format(scores[max_idx], c_candidates[max_idx]))

    def test_decision_trees_baseline(self, syn_fm, syn_labels, indices):

        features = syn_fm('2d')['0'][:]
        labels = syn_labels['0'][:]
        idx = indices['0']

        dt_classifier = DecisionTreeClassifier(random_state=seed)
        dt_classifier.fit(features[idx.train], labels[idx.train])
        score = dt_classifier.score(features[idx.test], labels[idx.test])

        probas = dt_classifier.predict_proba(features[idx.test])[:, 1]

        print("Decision Trees val_acc: {}".format(score))

    def test_linear_toy_classifier_baseline(self, syn_fm, syn_labels, indices):
        """
        Uses the same algorithm as LinearSVC, but with log loss instead of hinge loss.
        """
        features = syn_fm('2d')['0'][:]
        labels = syn_labels['0'][:]
        idx = indices['0']

        lr_classifier = LogisticRegression( C=Cs, penalty="l2", dual=True, verbose=0, random_state=seed)
        lr_classifier.fit(features[idx.train], labels[idx.train])
        assert (lr_classifier.classes_.shape[0] == 2)
        score = lr_classifier.score(features[idx.test], labels[idx.test])


        print("Logistic val_acc: {}".format(score))

    def test_dense_model_baseline(self, syn_fm, syn_labels, indices):
        params = {
            'epochs': 20,
        }

        features = syn_fm('2d')['0'][:]
        labels = syn_labels['0'][:]
        idx = indices['0']

        model = Sequential()
        model.add(Flatten(input_shape=(10020, 3)))

        model.add(Dense(activation='relu',
                        units=2,
                        ))

        model.add(Dense(activation='softmax',
                        units=2,
                        ))

        print(labels.train[:10])
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        model.summary()

        # Model fitting
        history = model.fit(x=features[idx.train], y=labels[idx.train],
                            validation_data=(features[idx.test], labels[idx.test]),
                            epochs=params['epochs'],
                            verbose=1)
        score = max(history.history['val_acc'])
        print("DNN val_acc: {}".format(score))

