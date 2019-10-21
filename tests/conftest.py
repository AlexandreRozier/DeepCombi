import os

import h5py
import numpy as np
import pytest
import scipy
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit

from Indices import Indices
from helpers import h5py_to_featmat, generate_syn_phenotypes
from parameters_complete import SYN_DATA_DIR, ttbr as default_ttbr, n_subjects, n_total_snps, noise_snps, inform_snps, \
    random_state, FINAL_RESULTS_DIR, REAL_DATA_DIR

TRAIN_PERCENTAGE = 0.80
TEST_PERCENTAGE = 0.20
VAL_PERCENTAGE = 1 - TRAIN_PERCENTAGE - TEST_PERCENTAGE


features_path = os.path.join(SYN_DATA_DIR, 'syn_data.h5py')

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
def real_pvalues():
    def real_pvalues_(disease, chromo):
        return np.load(os.path.join(FINAL_RESULTS_DIR, 'pvalues', disease, '{}.npy'.format(chromo)))
    return real_pvalues_


@pytest.fixture(scope="module")
def real_h5py_data():
    def real_data_(disease, chrom):
        return scipy.io.loadmat(os.path.join(REAL_DATA_DIR, disease, 'chromo_{}_processed.mat'.format(chrom)))['X']
        
    return real_data_

@pytest.fixture(scope="module")
def chrom_length():
    def chrom_length_(disease, chrom):
        try:
            _, shape, _ = scipy.io.whosmat(os.path.join(REAL_DATA_DIR, disease, 'chromo_{}.mat'.format(chrom)))[0][1] / 3.0

        except NotImplementedError:
            shape = h5py.File(os.path.join(REAL_DATA_DIR, disease, 'chromo_{}.mat'.format(chrom))).get('X').shape[0] / 3.0
        return int(shape)

    return chrom_length_




@pytest.fixture(scope='module')
def real_labels():
    def real_labels_(disease):
        try:
            return scipy.io.loadmat(os.path.join(REAL_DATA_DIR, disease, 'labels.mat'))['y'][0]

        except Exception as identifier:
            return h5py.File(os.path.join(REAL_DATA_DIR, disease, 'labels.mat'), 'r').get('y')[:].T[0]
    return real_labels_

@pytest.fixture(scope='module')
def real_labels_0based(real_labels):
    def real_labels_0based_(disease):
        return (real_labels(disease)+1)/2
    return real_labels_0based_

@pytest.fixture(scope='module')
def real_labels_cat(real_labels_0based):
    def real_labels_cat_(disease):
        return tf.keras.utils.to_categorical(real_labels_0based(disease))
    return real_labels_cat_

@pytest.fixture(scope='module')
def real_idx(real_h5py_data, real_labels_0based):
    def real_idx_(disease):
        n_subjects = len(real_labels_0based(disease))
        splitter =  StratifiedShuffleSplit(n_splits=1, test_size = TEST_PERCENTAGE, random_state=random_state)
        train_indices, test_indices = next(splitter.split(np.zeros(n_subjects), real_labels_0based(disease)))
        return Indices(train_indices, test_indices, None)   
    return real_idx_  

@pytest.fixture(scope='module')
def alphas():
    def alphas_(disease):

        with h5py.File(os.path.join(REAL_DATA_DIR, disease, 'alpha_j.mat', 'r')) as f:
            return f['alpha_j'].T[0]    
    return alphas_

@pytest.fixture(scope='module')
def alphas_EV():
    def alphas_EV_(disease):

        with h5py.File(os.path.join(REAL_DATA_DIR, disease, 'alpha_j_EV.mat', 'r')) as f:
            return f['alpha_j_EV'].T[0]   
    return alphas_EV_

@pytest.fixture(scope='function')
def labels(rep, ttbr):
    return generate_syn_phenotypes(ttbr=ttbr,
                                   root_path=SYN_DATA_DIR, n_info_snps=inform_snps, n_noise_snps=noise_snps, quantity=rep)
     
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



