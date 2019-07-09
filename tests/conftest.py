from keras.utils import to_categorical
from parameters_complete import DATA_DIR, ttbr as ttbr, seed, random_state ,n_subjects, n_total_snps, noise_snps, inform_snps, rep      
from helpers import string_to_featmat, generate_syn_phenotypes
from sklearn.model_selection import train_test_split
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
labels_path = os.path.join(DATA_DIR, 'syn_ttbr_{}_labels.h5py'.format(ttbr))

@pytest.fixture(scope="module")
def h5py_data():
    return h5py.File(features_path,'r')

@pytest.fixture(scope="module")
def fm(h5py_data):
    def fm_(dimension):
        return string_to_featmat(h5py_data, embedding_type=dimension)
    return fm_

@pytest.fixture(scope="module")
def labels():
    return generate_syn_phenotypes(ttbr=ttbr,
            root_path=DATA_DIR, n_info_snps=inform_snps, n_noise_snps=noise_snps)
     
@pytest.fixture(scope="module")
def labels_cat(labels):
    labels_cat = {}
    for key, l in labels.items():
        labels_cat[key] = to_categorical((l+1)/2)
    return labels_cat

@pytest.fixture(scope="module")
def labels_0based(labels):
    labels_0based = {}
    for key, l in labels.items():
        labels_0based[key] = ((l+1)/2).astype(int)
    return labels_0based


@pytest.fixture(scope="module")
def indices():
    indices_ = np.arange(n_subjects)
    np.random.shuffle(indices_)
    train_indices = indices_[:math.floor(n_subjects*TRAIN_PERCENTAGE)]
    test_indices = indices_[math.floor(
        n_subjects*TRAIN_PERCENTAGE):math.floor(n_subjects*(TRAIN_PERCENTAGE + TEST_PERCENTAGE))]
    val_indices = indices_[math.floor(n_subjects*(TRAIN_PERCENTAGE + TEST_PERCENTAGE)):]
    assert(len(np.intersect1d(train_indices,test_indices))==0)
    assert(len(np.intersect1d(train_indices,val_indices))==0)
    print('Dataset sizes: Train: {}; Test: {}; Validation: {}'.format(len(train_indices),len(test_indices), len(val_indices)))
    return Indices(train_indices, test_indices, val_indices)



#viz = Visdom(port=8097, server='http://localhost', base_url='/', username='', password='', \
#use_incoming_socket=True)

def pytest_addoption(parser):
    parser.addoption("--output_path", action="store", default="???")
    parser.addoption("--hparams_array_path", action="store", default="???")


@pytest.fixture
def hparams_array_path(request):
    return request.config.getoption("--hparams_array_path")

@pytest.fixture
def output_path(request):
    return request.config.getoption("--output_path")
