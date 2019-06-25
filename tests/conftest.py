import pytest
import tensorflow as tf
from parameters_complete import DATA_DIR, ttbr as ttbr, seed, random_state , n_total_snps, noise_snps, inform_snps, rep      
from keras.utils import to_categorical
from helpers import string_to_featmat, generate_syn_phenotypes
from sklearn.model_selection import train_test_split
import numpy as np
import os
import math
from time import time
import h5py

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
TRAIN_PERCENTAGE = 0.80
TEST_PERCENTAGE = 0.10
VAL_PERCENTAGE = 1 - TRAIN_PERCENTAGE - TEST_PERCENTAGE


features_path = os.path.join(DATA_DIR, 'syn_data.h5py')
labels_path = os.path.join(DATA_DIR, 'syn_ttbr_{}_labels.h5py'.format(ttbr))
print(features_path)


@pytest.fixture(scope="module")
def labels():   
    labels_ = generate_syn_phenotypes(ttbr=ttbr,
            root_path=DATA_DIR, n_info_snps=inform_snps, n_noise_snps=noise_snps)
    return labels_

@pytest.fixture(scope="module")
def h5py_data():
    return h5py.File(features_path,'r')


@pytest.fixture(scope="module")
def f_and_l(h5py_data, labels):

    def _f_and_l(embedding_type="3d", categorical=True):
        f_dict = {}
        l_dict = {}
        for key in list(h5py_data.keys()):
            if categorical==True:
                y_all = to_categorical((labels[key]+1)/2)
            else:
                y_all = labels[key]
            x_all = string_to_featmat(h5py_data[key][:], embedding_type=embedding_type)
            x_, x_test, y_, y_test = train_test_split(
                x_all, y_all, test_size=TEST_PERCENTAGE, random_state=random_state)
            x_train, x_val, y_train, y_val = train_test_split(
                x_, y_, test_size=math.floor(VAL_PERCENTAGE/(1-TEST_PERCENTAGE)),  random_state=random_state)

            l_dict[key] = Label(y_train, y_test, y_val, y_all)
            f_dict[key] = Features(x_train, x_test, x_val, x_all)
            print('Dataset sizes: Train: {}; Test: {}; Validation: {}'.format(len(x_train),len(x_test), len(x_val)))
        
        return f_dict, l_dict
    
    return _f_and_l





@pytest.fixture(scope="module")
def indices():
    with h5py.File(features_path,'r') as f, h5py.File(labels_path,'r') as l :
        n1 = f['X'].shape[0]
        n2 = l['X'].shape[0]
        assert n1 == n2

    indices_ = np.arange(n1)
    np.random.shuffle(indices_)
    train_indices = indices_[:math.floor(n1*TRAIN_PERCENTAGE)]
    test_indices = indices_[math.floor(
        n1*TRAIN_PERCENTAGE):math.floor(n1*(TRAIN_PERCENTAGE + TEST_PERCENTAGE))]
    val_indices = indices_[math.floor(n1*(TRAIN_PERCENTAGE + TEST_PERCENTAGE)):]
    assert(len(np.intersect1d(train_indices,test_indices))==0)
    assert(len(np.intersect1d(train_indices,val_indices))==0)
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
