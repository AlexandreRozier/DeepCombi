import h5py
import hdf5storage
import pytest
import scipy
import os
import numpy as np
from tqdm import tqdm
from helpers import generate_syn_genotypes,generate_syn_phenotypes, string_to_featmat, count_lines, check_genotype_unique_allels
from parameters_complete import (DATA_DIR, noise_snps, inform_snps)

class TestDataGeneration(object):
    
    
    n_replication = 33
    group_size = 300
    
    
    def test_synthetic_genotypes_generation(self, tmp_path):
        """ Generate synthetic genotypes
        """
        data_path = generate_syn_genotypes(root_path=DATA_DIR, n_replication=self.n_replication, 
                                        group_size=self.group_size, n_info_snps=inform_snps, n_noise_snps=noise_snps)
             
        with h5py.File(data_path, 'r') as file:
            print("Veryifying the generated phenotypes...")
            genotype = file['X'][:]
            n_indiv, n_snps, _ = genotype.shape
            assert n_indiv == self.n_replication*self.group_size
            assert n_snps == inform_snps + noise_snps
            check_genotype_unique_allels(genotype) # Checks that at most 3 unique allels exist

    def test_synthetic_phenotypes_generation(self, tmp_path):
        """ Generate synthetic phenotypes
        """
        labels_path = generate_syn_phenotypes(root_path=DATA_DIR, c=6, n_info_snps=inform_snps, n_noise_snps=noise_snps)
        lines = count_lines(labels_path) 
        assert(lines == self.n_replication*self.group_size)


    def test_feature_map_generation(self, tmp_path):
        with h5py.File(os.path.join(DATA_DIR, 'syn_data.txt'),'r') as d:
            raw_data = d['X'][:]
            fm = string_to_featmat(raw_data, embedding_type='2d', overwrite=True)
           