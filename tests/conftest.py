import os

import h5py
import numpy as np
import pytest
import scipy
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit

from Indices import Indices
from helpers import genomic_to_featmat, generate_syn_phenotypes
from parameters_complete import SYN_DATA_DIR, ttbr as default_ttbr, syn_n_subjects, n_total_snps, noise_snps, inform_snps, \
    random_state, FINAL_RESULTS_DIR, REAL_DATA_DIR

TRAIN_PERCENTAGE = 0.80
TEST_PERCENTAGE = 0.20
VAL_PERCENTAGE = 1 - TRAIN_PERCENTAGE - TEST_PERCENTAGE


features_path = os.path.join(SYN_DATA_DIR, 'genomic.h5py')

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
    """
    Returns the rower to base ratio

    """
    return int(request.config.getoption("--ttbr"))





@pytest.fixture(scope="module")
def chrom_length():
    """
    Returns the length of a specific chromosom of a specific disease
    :return:
    """
    def chrom_length_(disease, chrom):
        try:
            _, shape, _ = scipy.io.whosmat(os.path.join(REAL_DATA_DIR, disease, 'chromo_{}.mat'.format(chrom)))[0][1] / 3.0

        except NotImplementedError:
            shape = h5py.File(os.path.join(REAL_DATA_DIR, disease, 'chromo_{}.mat'.format(chrom))).get('X').shape[0] / 3.0
        return int(shape)

    return chrom_length_




@pytest.fixture(scope='module')
def alphas():
    def alphas_(disease):

        with h5py.File(os.path.join(REAL_DATA_DIR, disease, 'alpha_j.mat', 'r')) as f:
            return f['alpha_j'].T[0]    
    return alphas_

@pytest.fixture(scope='module')
def alphas_ev():
    def alphas_EV_(disease):

        with h5py.File(os.path.join(REAL_DATA_DIR, disease, 'alpha_j_EV.mat', 'r')) as f:
            return f['alpha_j_EV'].T[0]   
    return alphas_EV_

"""
++++++++++++ TOY/SYNTHETIC DATA PROVIDERS +++++++++++=
"""

@pytest.fixture(scope='function')
def syn_true_pvalues(rep):
    """
    An array of zeroes except for where the loci are informative, in case 1.
    """
    pvalues = np.zeros((rep, n_total_snps), dtype=bool)
    pvalues[:, int(noise_snps/2):int(noise_snps/2)+inform_snps] = True
    return pvalues

@pytest.fixture(scope="module")
def syn_genomic_data():
    """
    Provides the 3D genomic data corresponding to all synthetic datasets.
    {
        '0': matrix(N, n_snps, 2),
        '1': matrix(N, n_snps, 2),
        ...,
        'rep':matrix(N, n_snps, 2)
    }
    -> where 2 corresponds to the number of allels.
    """
    return h5py.File(features_path,'r')

@pytest.fixture(scope="module")
def syn_fm(syn_genomic_data):
    """
    Returns a dictionary of feature matrices associated to the synthetic genomic datasets
    :return:
    """
    def fm_(dimension):
        return genomic_to_featmat(embedding_type=dimension)
    return fm_


@pytest.fixture(scope='function')
def syn_labels(rep, ttbr):
    """
    Loads synthetic labels for all datasets
    :return:
    {
        '0':[-1,-1,...,1],
        ...,
        'rep':[1,1,-1,...,-1]
    }

    """
    return generate_syn_phenotypes(root_path=SYN_DATA_DIR, tower_to_base_ratio=ttbr, n_info_snps=inform_snps, n_noise_snps=noise_snps,
                                   quantity=rep)
     
@pytest.fixture(scope='function')
def syn_labels_cat(syn_labels):
    """ Same as syn_labels but labels are either 0 or 1
    """

    labels_cat = {}
    for key, l in syn_labels.items():
        labels_cat[key] = tf.keras.utils.to_categorical((l+1)/2)
    return labels_cat

@pytest.fixture(scope='function')
def syn_labels_0based(syn_labels):
    """ Same as syn_labels but labels are 2-hot-encoded
    """
    labels_0based = {}
    for key, l in syn_labels.items():
        labels_0based[key] = ((l+1)/2).astype(int)
    return labels_0based


@pytest.fixture(scope="function")
def syn_idx(syn_labels_0based):
    """ Gets indices splitting our datasets into  train and test sets
    :return
    {
        '0':Index,
        ...,
        'rep':Index
    }
    Ex: indices['0'].train gets the indices for dataset 0, train set.
    """
    assert VAL_PERCENTAGE <=0.00001
    indices_ = {}
    splitter =  StratifiedShuffleSplit(n_splits=1, test_size = TEST_PERCENTAGE, random_state=random_state)
    for key, labels in syn_labels_0based.items():

        train_indices, test_indices = next(splitter.split(np.zeros(syn_n_subjects), labels))
        indices_[key] = Indices(train_indices, test_indices, None)
    
    print('Dataset sizes: Train: {}; Test: {}; Validation: ERROR'.format(len(train_indices),len(test_indices)))
    return indices_

"""
++++++++++++ WTCCC DATA PROVIDERS +++++++++++=
"""

@pytest.fixture(scope='module')
def real_labels():

    """
    Loads real labels given a specific disease
    :return: the array of -1, +1 encoded labels
    """
    def real_labels_(disease):
        try:
            return scipy.io.loadmat(os.path.join(REAL_DATA_DIR, disease, 'labels.mat'))['y'][0]

        except Exception:
            return h5py.File(os.path.join(REAL_DATA_DIR, disease, 'labels.mat'), 'r').get('y')[:].T[0]
    return real_labels_

@pytest.fixture(scope='module')
def real_labels_0based(real_labels):
    """
    Loads real labels given a specific disease
    :return: the array of 0, +1 encoded labels
    """
    def real_labels_0based_(disease):
        return (real_labels(disease)+1)/2
    return real_labels_0based_

@pytest.fixture(scope='module')
def real_labels_cat(real_labels_0based):
    """
    Loads real labels given a specific disease
    :return: the array of 2-hot-encoded labels
    """
    def real_labels_cat_(disease):
        return tf.keras.utils.to_categorical(real_labels_0based(disease))
    return real_labels_cat_

@pytest.fixture(scope='module')
def real_idx(real_genomic_data, real_labels_0based):
    """
    Loads indices splitting subjects into a train a test set
    :return: an Index object (for instance, access train indices via Index.train)
    """
    def real_idx_(disease):
        n_subjects = len(real_labels_0based(disease))
        splitter =  StratifiedShuffleSplit(n_splits=1, test_size = TEST_PERCENTAGE, random_state=random_state)
        train_indices, test_indices = next(splitter.split(np.zeros(n_subjects), real_labels_0based(disease)))
        return Indices(train_indices, test_indices, None)
    return real_idx_


@pytest.fixture(scope="module")
def real_pvalues():
    """
    Loads pvalues computed for a specific disease, for a specific chromosom
    :return: an Index object (for instance, access train indices via Index.train)
    """
    def real_pvalues_(disease, chromo):
        return np.load(os.path.join(REAL_DATA_DIR, 'pvalues', disease, '{}.npy'.format(chromo)))

    return real_pvalues_


@pytest.fixture(scope="module")
def real_genomic_data():
    """
    Provides the 3D genomic data corresponding to disease, chromosome
    Shape: (N, n_snps, 2) -> 2 corresponds to the number of allels.
    """
    def real_data_(disease, chrom):
        return scipy.io.loadmat(os.path.join(REAL_DATA_DIR, disease, 'chromo_{}_processed.mat'.format(chrom)))['X']

    return real_data_
