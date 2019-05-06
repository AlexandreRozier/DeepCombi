import pytest
from keras.utils import to_categorical
from helpers import chi_square, permuted_combi, string_to_featmat
from sklearn.model_selection import train_test_split
import numpy as np
import os


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


VAL_PERCENTAGE = 0.25
# Test size = dataset_size * (1-VAL_PERCENTAGE)*TEST_PERCENTAGE
TEST_PERCENTAGE = 0.33
skiprows = 0
x_train = None
x_test = None
x_val = None
x_all = None
y_train = None
y_test = None
y_val = None
y_all = None

seed = 666

with open('./data/data.txt', 'r') as d, open('./data/labels.txt', 'r') as l:
    my_labels = np.loadtxt(l, dtype=np.int8, skiprows=skiprows)
    y_all = to_categorical(my_labels)
    x_all = string_to_featmat(np.loadtxt(
        d, np.chararray, skiprows=skiprows), embedding_type='2d')  # [:,4800*3:5200*3]
    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_all, test_size=VAL_PERCENTAGE, random_state=np.random.RandomState(seed))
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=TEST_PERCENTAGE,  random_state=np.random.RandomState(seed))
    print("FIXTURES SIZES: train: {}; test: {}; val:{}".format(
        x_train.shape[0], x_test.shape[0], x_test.shape[0]))


@pytest.fixture(scope = "module")
def labels():
    return Label(y_train, y_test, y_val, y_all)


@pytest.fixture(scope = "module")
def features():
    return Features(x_train, x_test, x_val, x_all)


@pytest.fixture(scope = "module")
def test_dir():
    return os.path.dirname(os.path.realpath(__file__))

@pytest.fixture(scope = "module")
def data_dir():
    return os.path.dirname(os.path.realpath(__file__)) + '/../data/'


@pytest.fixture(scope = "module")
def talos_output_dir():
    return os.path.dirname(os.path.realpath(__file__)) + "/talos_output/"


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
