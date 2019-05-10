import pytest
from parameters_complete import DATA_DIR
from keras.utils import to_categorical
from helpers import string_to_featmat
from sklearn.model_selection import train_test_split
import numpy as np
import os
from time import time
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


# Test size = dataset_size * TEST_PERCENTAGE = 20%
# Train size = dataset_size * (1-TEST_PERCENTAGE) * (1-VAL_PERCENTAGE) = 64 %
# Val size = 16%
VAL_PERCENTAGE = 0.20
TEST_PERCENTAGE = 0.20
skiprows = 0
x_train = None
x_test = None
x_val = None
x_all = None
y_train = None
y_test = None
y_val = None
y_all = None

raw_data = None
raw_labels = None
seed = 666



with open(os.path.join(DATA_DIR,'syn_data.txt'), 'r') as d, open(os.path.join(DATA_DIR,'syn_labels.txt'), 'r') as l:
    print('STARTING DATA PARSING...')
    start_time = time()
    raw_labels = np.loadtxt(l, dtype=np.int8, skiprows=skiprows)
    raw_data = np.loadtxt(d, np.chararray, skiprows=skiprows)
    y_all = to_categorical(raw_labels)
    x_all = string_to_featmat(raw_data, embedding_type='2d')
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=TEST_PERCENTAGE, random_state=np.random.RandomState(seed))
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=VAL_PERCENTAGE,  random_state=np.random.RandomState(seed))
    print("FIXTURES SIZES: train: {}; test: {}; val:{}".format(
        x_train.shape[0], x_test.shape[0], x_test.shape[0]))
    print("TIME ELASPED DURING DATA PARSING:{}".format(time()-start_time))


@pytest.fixture(scope = "module")
def raw_data():
    return raw_data

@pytest.fixture(scope = "module")
def raw_labels():
    return raw_labels


@pytest.fixture(scope = "module")
def labels():
    return Label(y_train, y_test, y_val, y_all)


@pytest.fixture(scope = "module")
def features():
    return Features(x_train, x_test, x_val, x_all)





def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent=item.parent
            parent._previousfailed=item


def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed=getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("previous test failed (%s)" % previousfailed.name)
