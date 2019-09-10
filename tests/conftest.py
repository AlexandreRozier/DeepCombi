from keras.utils import to_categorical
from parameters_complete import DATA_DIR, ttbr as default_ttbr, seed, random_state ,n_subjects, n_total_snps, noise_snps, inform_snps      , random_state
from helpers import h5py_to_featmat, generate_syn_phenotypes
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
import os
import math
from time import time
import h5py
from tqdm import tqdm
import pytest


from Indices import Indices


TRAIN_PERCENTAGE = 0.80
TEST_PERCENTAGE = 0.20
VAL_PERCENTAGE = 1 - TRAIN_PERCENTAGE - TEST_PERCENTAGE


features_path = os.path.join(DATA_DIR, 'syn_data.h5py')

# Fancy command-line options
def pytest_addoption(parser):
    parser.addoption("--output_path", action="store", default="/tmp")
    parser.addoption("--hparams_array_path", action="store", default="???")
    parser.addoption("--rep", action="store", default=2)
    parser.addoption("--ttbr", action="store", default=default_ttbr)


@pytest.fixture
def output_path(request):
    return request.config.getoption("--output_path")

@pytest.fixture(scope='function')
def rep(request):
    return int(request.config.getoption("--rep"))

@pytest.fixture(scope='function')
def ttbr(request):
    return int(request.config.getoption("--ttbr"))

@pytest.fixture(scope='function')
def true_pvalues(rep):
    pvalues = np.zeros((rep, n_total_snps), dtype=bool)
    pvalues[:, int(noise_snps/2):int(noise_snps/2)+inform_snps] = True
    return pvalues


@pytest.fixture(scope="module")
def h5py_data():
    return h5py.File(features_path,'r')

@pytest.fixture(scope="module")
def fm(h5py_data):
    def fm_(dimension):
        return h5py_to_featmat(h5py_data, embedding_type=dimension)
    return fm_

@pytest.fixture(scope='function')
def labels(rep, ttbr):
    return generate_syn_phenotypes(ttbr=ttbr,
            root_path=DATA_DIR, n_info_snps=inform_snps, n_noise_snps=noise_snps,quantity=rep)
     
@pytest.fixture(scope='function')
def labels_cat(labels):
    labels_cat = {}
    for key, l in labels.items():
        labels_cat[key] = to_categorical((l+1)/2)
    return labels_cat

@pytest.fixture(scope='function')
def labels_0based(labels):
    labels_0based = {}
    for key, l in labels.items():
        labels_0based[key] = ((l+1)/2).astype(int)
    return labels_0based


@pytest.fixture(scope="function")
def indices(labels_0based):
    """ Gets stratified indices. WARNING not usable with a val set
    """
    assert VAL_PERCENTAGE <=0.00001
    indices_ = {}
    splitter =  StratifiedShuffleSplit(n_splits=1, test_size = TEST_PERCENTAGE, random_state=random_state)
    for key, labels in labels_0based.items():

        train_indices, test_indices = next(splitter.split(np.zeros(n_subjects), labels))
        indices_[key] = Indices(train_indices, test_indices, None)
    
    print('Dataset sizes: Train: {}; Test: {}; Validation: ERROR'.format(len(train_indices),len(test_indices)))
    return indices_



