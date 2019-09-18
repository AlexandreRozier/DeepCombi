import tensorflow as tf
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
import scipy


from Indices import Indices


TRAIN_PERCENTAGE = 0.80
TEST_PERCENTAGE = 0.20
VAL_PERCENTAGE = 1 - TRAIN_PERCENTAGE - TEST_PERCENTAGE


features_path = os.path.join(DATA_DIR, 'syn_data.h5py')

# Fancy command-line options
def pytest_addoption(parser):
    parser.addoption("--output_path", action="store", default="/tmp")
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


@pytest.fixture(scope="module")
def real_h5py_data():
    def real_data_(chrom):
        f = h5py.File(os.path.join(DATA_DIR,'chromo_{}.mat'.format(chrom)),'r')
        data = f.get('X')[:].T            
        print('Chromosom {} takes {} GB of memory'.format(chrom, data.nbytes/(1024*1024*1024)))                                                     
        return data.reshape(data.shape[0],-1,3)[:, :, :2] 
        
    return real_data_

@pytest.fixture(scope='module')
def real_labels():
    return scipy.io.loadmat(os.path.join(DATA_DIR,'labels.mat'))['y'][0]
    
@pytest.fixture(scope='module')
def real_labels_0based(real_labels):
    return (real_labels+1)/2

@pytest.fixture(scope='module')
def real_labels_cat(real_labels_0based):
    return tf.keras.utils.to_categorical(real_labels_0based)

@pytest.fixture(scope='module')
def real_idx(real_h5py_data, real_labels_0based):
    n_subjects = len(real_labels_0based)
    splitter =  StratifiedShuffleSplit(n_splits=1, test_size = TEST_PERCENTAGE, random_state=random_state)
    train_indices, test_indices = next(splitter.split(np.zeros(n_subjects), real_labels_0based))
    return Indices(train_indices, test_indices, None)     

@pytest.fixture(scope='module')
def alphas():
    return scipy.io.loadmat(os.path.join(DATA_DIR,'alpha_j.mat'))['alpha_j'].T[0]
    
@pytest.fixture(scope='module')
def alphas_EV():
    return scipy.io.loadmat(os.path.join(DATA_DIR,'alpha_j_EV.mat'))['alpha_j_EV'].T[0]


@pytest.fixture(scope='function')
def labels(rep, ttbr):
    return generate_syn_phenotypes(ttbr=ttbr,
            root_path=DATA_DIR, n_info_snps=inform_snps, n_noise_snps=noise_snps,quantity=rep)
     
@pytest.fixture(scope='function')
def labels_cat(labels):
    labels_cat = {}
    for key, l in labels.items():
        labels_cat[key] = tf.keras.utils.to_categorical((l+1)/2)
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



