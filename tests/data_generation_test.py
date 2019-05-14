import h5py
import hdf5storage
import pytest
import scipy
import os
import numpy as np
from tqdm import tqdm
from helpers import generate_syn_genotypes, string_to_featmat
from parameters_complete import DATA_DIR

class TestDataGeneration(object):
    
    

    def test_synthetic_generation(self, tmp_path):

        n_replication = 15
        group_size = 300
        n_info_snps = 20
        n_noise_snps = 10000
        data_path, _ = generate_syn_genotypes(root_path=DATA_DIR,c=6, n_replication=n_replication, group_size=group_size, n_info_snps=20, n_noise_snps=10000)
             
        with open(data_path,'r') as d:
            raw_data = np.loadtxt(d, np.chararray)
            n_indiv, n_snps = raw_data.shape

            assert n_indiv == n_replication*group_size
            assert n_snps == n_info_snps + n_noise_snps


    def test_feature_map_generation(self, tmp_path):
        with open(os.path.join(DATA_DIR, 'syn_data.txt'),'r') as d:
            raw_data = np.loadtxt(d, np.chararray)
            string_to_featmat(raw_data, embedding_type='3d', overwrite=True)
