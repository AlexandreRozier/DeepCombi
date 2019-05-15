import pytest
from parameters_complete import DATA_DIR
from keras.utils import to_categorical
from helpers import string_to_featmat
from sklearn.model_selection import train_test_split
import numpy as np
import os
import math
from time import time
from helpers import count_lines


class Label:
    def __init__(self, train, test, val, all):
        self.test = test
        self.train = train
        self.val = val
        self.all = all


class Features:
    def __init__(self, train, test, val, all):
        self.test = test
        self.train = train
        self.val = val
        self.all = all


class Indices:
    def __init__(self, train, test, val):
        self.train = train
        self.test = test
        self.val = val


# Train on 40% of data, test on 10%
# So that 5-Fold validation computes results for the same train/test sizes
TRAIN_PERCENTAGE = 0.40
TEST_PERCENTAGE = 0.10
VAL_PERCENTAGE = 1 - TRAIN_PERCENTAGE - TEST_PERCENTAGE
seed = 666
np.random.seed(seed)
random_state = np.random.RandomState(seed)
skiprows = 0

features_path = os.path.join(DATA_DIR, 'syn_data.txt')
labels_path = os.path.join(DATA_DIR, 'syn_labels.txt')


@pytest.fixture(scope="module")
def raw_labels():   
    with open(labels_path, 'r') as l:
        return np.loadtxt(l, dtype=np.int8, skiprows=skiprows)


@pytest.fixture(scope="module")
def raw_data():
    with open(features_path, 'r') as d:
        return np.loadtxt(d, np.chararray, skiprows=skiprows)


@pytest.fixture(scope="module")
def f_and_l(raw_data, raw_labels):
    y_all = to_categorical(raw_labels)
    x_all = string_to_featmat(raw_data, embedding_type='3d')
    x_, x_test, y_, y_test = train_test_split(
        x_all, y_all, test_size=TEST_PERCENTAGE, random_state=random_state)
    x_train, x_val, y_train, y_val = train_test_split(
        x_, y_, test_size=math.floor(VAL_PERCENTAGE/(1-TEST_PERCENTAGE)),  random_state=random_state)

    print('Dataset sizes: Train: {}; Test: {}; Validation: {}'.format(len(x_train),len(x_test), len(x_val)))
    return {
        'features':Features(x_train, x_test, x_val, x_all), 
        'labels':Label(y_train, y_test, y_val, y_all)
    }




@pytest.fixture(scope="module")
def indices():
    n1 = count_lines(features_path)-1  # Don't forget the extra \n line
    n2 = count_lines(labels_path)-1
    assert n1 == n2

    indices_ = np.arange(n1)
    np.random.shuffle(indices_)
    train_indices = indices_[:math.floor(n1*TRAIN_PERCENTAGE)]
    test_indices = indices_[math.floor(
        n1*TRAIN_PERCENTAGE):math.floor(n1*(TRAIN_PERCENTAGE + TEST_PERCENTAGE))]
    val_indices = indices_[math.floor(n1*(TRAIN_PERCENTAGE + TEST_PERCENTAGE)):]

    print('Dataset sizes: Train: {}; Test: {}; Validation: {}'.format(len(train_indices),len(test_indices), len(val_indices)))
    return Indices(train_indices, test_indices, val_indices)


def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("previous test failed (%s)" % previousfailed.name)
