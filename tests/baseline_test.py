import os
import pytest
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l1
from parameters_complete import (Cs, PARAMETERS_DIR, DATA_DIR, seed)
from models import DataGenerator
from helpers import EnforceNeg, generate_name_from_params


@pytest.mark.incremental
class TestBaselines(object):

    CONF_PATH = os.path.join(PARAMETERS_DIR, os.environ['SGE_TASK_ID'])

    def test_svm_baseline(self, f_and_l):
        """
        SVM needs 2D data and does not support one hot encoded labels
        """

        features, labels = f_and_l(embedding_type="2d", categorical=False)

        # More features than datapoints
        classifier = svm.LinearSVC(
            C=Cs, penalty='l2', verbose=1, dual=True, random_state=seed)
        classifier.fit(features.train, labels.train )
        score = classifier.score(features.test, labels.test )

        print("SVM score: {}".format(score))

    def test_decision_trees_baseline(self, f_and_l):

        features, labels = f_and_l(embedding_type="2d", categorical=False)

        classifier = DecisionTreeClassifier(random_state=seed)
        classifier.fit(features.train, labels.train)
        score = classifier.score(features.test, labels.test)

        probas = classifier.predict_proba(features.test)[:, 1]
        roc_auc = roc_auc_score(labels.test, probas)

        print("Decision Trees score: {}; ROC Auc: {}".format(score, roc_auc))

    def test_linear_classifier_baseline(self, f_and_l):
        """
        Uses the same algorithm as LinearSVC, but with log loss instead of hinge loss.
        """
        features, labels = f_and_l(embedding_type="2d", categorical=False)

        classifier = LogisticRegression(
            C=Cs, penalty="l2", dual=True, verbose=1, random_state=seed)
        classifier.fit(features.train, labels.train)
        assert(classifier.classes_.shape[0] == 2)
        score = classifier.score(features.test, labels.test)

        probas = classifier.predict_proba(features.test)[:, 1]
        roc_auc = roc_auc_score(labels.test, probas)

        print("Logistic regr score: {}; ROC Auc: {}".format(score, roc_auc))

    def test_dense_model_baseline(self, f_and_l):
        params = {
            'epochs': 500,
            'batch_size': 500,
            'verbose': 1,
        }

        features, labels = f_and_l(embedding_type="3d", categorical=False)
        print(labels.train[:100])
        model = Sequential()
        model.add(Flatten(input_shape=(10020, 3)))

        model.add(Dense(units=1,
                        activation='tanh',
                        #kernel_regularizer=l1(0.1),
                        #bias_constraint=EnforceNeg()))  # Negative bias are crucial for LRP
                        ))
        model.compile(loss='binary_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])

        model.summary()

        early_stopping_cb = EarlyStopping(
            monitor='val_loss',  patience=8,  baseline=None)
        reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss')
        # Model fitting
        history = model.fit(x=features.train, y=(labels.train +1) /2,
                            validation_data=(features.test, (labels.test +1) /2),
                            epochs=params['epochs'],
                            verbose=params['verbose'],
                            callbacks=[
                                # early_stopping_cb,
                                #reduce_lr_cb
                            ])
        score = max(history.history['val_acc'])
        print("DNN score: {}".format(score))
