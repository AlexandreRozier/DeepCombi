import os
import pytest
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from parameters_complete import (Cs, PARAMETERS_DIR, seed)

@pytest.mark.incremental
class TestBaselines(object):

    CONF_PATH = os.path.join(PARAMETERS_DIR,os.environ['SGE_TASK_ID'])

    def test_svm_baseline(self, f_and_l):
        """
        SVM needs 2D data and does not support one hot encoded labels
        """

        features, labels = f_and_l(embedding_type="2d", categorical=False)
            
        classifier = svm.LinearSVC(C=Cs, penalty='l2', verbose=1, dual=True, random_state=seed) # More features than datapoints
        classifier.fit(features.train, labels.train)        
        score = classifier.score(features.test, labels.test)
        
        print("SVM score: {}".format(score))


    def test_decision_trees_baseline(self, f_and_l):


        features, labels = f_and_l(embedding_type="2d", categorical=False)
            
        classifier = DecisionTreeClassifier(random_state=seed)
        classifier.fit(features.train, labels.train)        
        score = classifier.score(features.test, labels.test)

        probas = classifier.predict_proba(features.test)[:,1]
        roc_auc = roc_auc_score(labels.test, probas)        

        print("Decision Trees score: {}; ROC Auc: {}".format(score, roc_auc))


    def test_linear_classifier_baseline(self, f_and_l):
        """
        Uses the same algorithm as LinearSVC, but with log loss instead of hinge loss.
        """
        features, labels = f_and_l(embedding_type="2d", categorical=False)
            
        classifier = LogisticRegression(C=Cs, penalty="l2",dual=True, verbose=1,random_state=seed)
        classifier.fit(features.train, labels.train)      
        assert(classifier.classes_.shape[0]==2)
        score = classifier.score(features.test, labels.test)

        probas = classifier.predict_proba(features.test)[:,1]
        roc_auc = roc_auc_score(labels.test, probas)       

        print("Logistic regr score: {}; ROC Auc: {}".format(score, roc_auc))
