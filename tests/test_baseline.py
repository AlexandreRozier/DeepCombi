import os
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import tensorflow as tensorflow
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l1
from parameters_complete import (Cs, PARAMETERS_DIR, DATA_DIR,TEST_DIR, n_total_snps, seed)
from models import DataGenerator
from helpers import EnforceNeg, generate_name_from_params

from tqdm import tqdm
import innvestigate
import innvestigate.utils as iutils
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

class TestBaselines(object):

    CONF_PATH = os.path.join(PARAMETERS_DIR, os.environ['SGE_TASK_ID'])



    def test_svm_baseline(self, f_and_l):
        """
        SVM needs 2D data and does not support one hot encoded labels
        """

        features, labels = f_and_l(embedding_type="2d", categorical=False)
        c_candidates = np.logspace(-5,1, 20) 
        # More features than datapoints
        val_scores = np.zeros(len(c_candidates))
        train_scores = np.zeros(len(c_candidates))
        for i,c in tqdm(enumerate(c_candidates)):

            classifier = svm.LinearSVC(C=c, penalty='l2', verbose=0, dual=True, 
                                        random_state=seed)
            classifier.fit(features.train, labels.train )
            train_scores[i] = classifier.score(features.train, labels.train )
            val_scores[i] = classifier.score(features.test, labels.test )

        max_idx = np.argmax(val_scores)
        print("SVM train_acc: {} (std={}) ; obtained with C={} ".format(train_scores[max_idx],np.std(train_scores), train_scores[max_idx]))
        print("SVM val_acc: {} (std={}) ; obtained with C={}".format(val_scores[max_idx],np.std(val_scores), c_candidates[max_idx]))

    def test_kernel_svm_baseline(self, f_and_l):
        """
        SVM needs 2D data and does not support one hot encoded labels
        """

        features, labels = f_and_l(embedding_type="2d", categorical=False)
        c_candidates = np.logspace(-5,0, 10) 
                
        scores = np.zeros(len(c_candidates))
        for i,c in tqdm(enumerate(c_candidates)):

            classifier = svm.SVC(C=c, kernel='rbf', verbose=0, random_state=seed)
            classifier.fit(features.train, labels.train )
            scores[i] = classifier.score(features.test, labels.test )

        max_idx = np.argmax(scores)
        
        
        print("Kernel SVM val_acc: {} obtained with C={}".format(scores[max_idx], c_candidates[max_idx]))



    def test_decision_trees_baseline(self, f_and_l):

        features, labels = f_and_l(embedding_type="2d", categorical=False)

        classifier = DecisionTreeClassifier(random_state=seed)
        classifier.fit(features.train, labels.train)
        score = classifier.score(features.test, labels.test)

        probas = classifier.predict_proba(features.test)[:, 1]
        roc_auc = roc_auc_score(labels.test, probas)

        print("Decision Trees val_acc: {}; ROC Auc: {}".format(score, roc_auc))

    def test_linear_classifier_baseline(self, f_and_l):
        """
        Uses the same algorithm as LinearSVC, but with log loss instead of hinge loss.
        """
        features, labels = f_and_l(embedding_type="2d", categorical=False)

        classifier = LogisticRegression(
            C=Cs, penalty="l2", dual=True, verbose=0, random_state=seed)
        classifier.fit(features.train, labels.train)
        assert(classifier.classes_.shape[0] == 2)
        score = classifier.score(features.test, labels.test)

        probas = classifier.predict_proba(features.test)[:, 1]
        roc_auc = roc_auc_score(labels.test, probas)

        print("Logistic val_acc: {}; ROC Auc: {}".format(score, roc_auc))

    def test_dense_model_baseline(self, f_and_l):
        params = {
            'epochs': 20,
        }

        features, labels = f_and_l(embedding_type="3d", categorical=True)
        features = features['0']
        labels = labels['0']
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
        history = model.fit(x=features.train, y=labels.train ,
                            validation_data=(features.test, labels.test ),
                            epochs=params['epochs'],
                            verbose=1)
        score = max(history.history['val_acc'])
        print("DNN val_acc: {}".format(score))

        model_wo_sm = iutils.keras.graph.model_wo_softmax(model)
        model_wo_sm.summary()
       
        # Creating an analyzer
        
        inf = 0
        sup = 10020
        x = range(inf, sup)
        f, (ax1,ax2,ax3, ax4) = plt.subplots(4)
        dtaylor_analyzer = innvestigate.analyzer.LRPZ(model_wo_sm)
        analysis = dtaylor_analyzer.analyze(features.all).sum(axis=2)
        analysis = analysis[:,inf:sup]
        ax1.scatter(x, analysis.sum(0), marker='x', alpha=0.7, label="LRPZ")
        ax1.axvspan(5001, 5020, alpha=0.5, color='red')
        ax1.legend(loc="upper right")

        dtaylor_analyzer = innvestigate.analyzer.LRPZPlus(model_wo_sm)
        analysis = dtaylor_analyzer.analyze(features.all).sum(axis=2)
        analysis = analysis[:,inf:sup]
        ax2.scatter(x, analysis.sum(0), marker='x', alpha=0.7, label='LRPZPlus')
        ax2.axvspan(5001, 5020, alpha=0.5, color='red')
        ax2.legend(loc="upper right")

        dtaylor_analyzer = innvestigate.analyzer.DeepTaylor(model_wo_sm, {'low':-1,'high':1})
        analysis = dtaylor_analyzer.analyze(features.all).sum(axis=2)
        analysis = analysis[:,inf:sup]
        ax3.scatter(x, analysis.sum(0), marker='x', alpha=0.7, label='DeepTaylor')
        ax3.axvspan(5001, 5020, alpha=0.5, color='red')
        ax3.legend(loc="upper right")

        dtaylor_analyzer = innvestigate.analyzer.Gradient(model_wo_sm)
        analysis = dtaylor_analyzer.analyze(features.all).sum(axis=2)
        analysis = analysis[:,inf:sup]
        ax4.scatter(x, analysis.sum(0), marker='x', alpha=0.7, label='Gradient')
        ax4.axvspan(5001, 5020, alpha=0.5, color='red')
        ax4.legend(loc="upper right")

        f.savefig(os.path.join(TEST_DIR,'lrp.png'))
