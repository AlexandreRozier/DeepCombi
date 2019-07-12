import h5py
import scipy
import os
import numpy as np
from tqdm import tqdm
from helpers import generate_syn_genotypes, generate_syn_phenotypes, string_to_featmat, check_genotype_unique_allels
from parameters_complete import (
    DATA_DIR, noise_snps, inform_snps, n_total_snps, n_subjects, ttbr as ttbr) 


class TestDataGeneration(object):
    pass
    
        
    def test_syntest_text_to_hdf5(self):

        filename = 'data/bett_data.txt'
        lines_nb = sum(1 for line in open(filename))
        data = np.zeros((lines_nb, 10020, 2))
        with open(filename,'rb') as f:
            for i, line in enumerate(f):
                line = np.frombuffer(line, dtype='uint8')[:-1]
                line = line.reshape(-1,3)[:,:2]
                data[i] = line
            assert i == lines_nb - 1
        
        labels = np.loadtxt('data/bett_labels.txt')

        with h5py.File('data/bett_data.h5py', 'w') as file:
            file.create_dataset("X", data=data)
            
        with h5py.File('data/bett_labels.h5py', 'w') as file:
            file.create_dataset("X", data=labels)
    

    def test_synthetic_genotypes_generation(self, rep):
        
        data_path = generate_syn_genotypes(root_path=DATA_DIR, n_subjects=n_subjects,
                                           n_info_snps=inform_snps, n_noise_snps=noise_snps, 
                                           quantity=rep)

        with h5py.File(data_path, 'r') as file:
            print("Veryifying the generated phenotypes...")
            genotype = file['0'][:]
            n_indiv, n_snps, _ = genotype.shape
            assert n_indiv == n_subjects
            assert n_snps == inform_snps + noise_snps
            # Checks that at most 3 unique allels exist
            check_genotype_unique_allels(genotype)

    def test_synthetic_phenotypes_generation(self, rep):
       
        labels = generate_syn_phenotypes(ttbr=ttbr,
            root_path=DATA_DIR, n_info_snps=inform_snps, n_noise_snps=noise_snps, quantity=rep)['0']
        assert(labels.shape[0] == n_subjects)

    
    def test_proportion_of_labels(self, rep):
        
        labels = generate_syn_phenotypes(ttbr=ttbr,
            root_path=DATA_DIR, n_info_snps=inform_snps, n_noise_snps=noise_snps, quantity=rep)['0']
        print(sum(labels == 1))
        print(sum(labels == -1))

    def test_feature_map_generation(self):
        with h5py.File(os.path.join(DATA_DIR, 'syn_data.h5py'), 'r') as d:
            raw_data = d['0'][:]
            fm2 = string_to_featmat(
                raw_data, embedding_type='2d', overwrite=True)
            fm3 = string_to_featmat(
                raw_data, embedding_type='3d', overwrite=True)
            assert fm2['0'].shape[0] == n_subjects
            assert fm3['0'].shape[0] == n_subjects
            assert fm2['0'].shape[1] == n_total_snps * 3
            assert fm3['0'].shape[1] == n_total_snps

