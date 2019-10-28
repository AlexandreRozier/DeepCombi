import os

import h5py
import numpy as np
import scipy
from tqdm import tqdm

from helpers import generate_syn_genotypes, generate_syn_phenotypes, h5py_to_featmat, check_genotype_unique_allels, \
    chi_square
from parameters_complete import (
    SYN_DATA_DIR, noise_snps, inform_snps, n_total_snps, syn_n_subjects, ttbr as ttbr, disease_IDs, FINAL_RESULTS_DIR)


class TestDataGeneration(object):
    """
    This class takes care of generating all synthetic and real data & labels
    """
    
        
    def test_syntest_text_to_hdf5(self):

        filename = 'data/bett_data.txt'
        lines_nb = sum(1 for _ in open(filename))
        data = np.zeros((lines_nb, 10020, 2))
        with open(filename,'rb') as f:
            for i, line in enumerate(f):
                line = np.frombuffer(line, dtype='uint8')[:-1]
                line = line.reshape(-1,3)[:,:2]
                data[i] = line
            assert i == lines_nb - 1
        

        with h5py.File('data/bett_data.h5py', 'w') as file:
            file.create_dataset("X", data=data)
            

    

    def test_synthetic_genotypes_generation(self, rep):
        
        data_path = generate_syn_genotypes(root_path=SYN_DATA_DIR, n_subjects=syn_n_subjects,
                                           n_info_snps=inform_snps, n_noise_snps=noise_snps,
                                           quantity=rep)

        with h5py.File(data_path, 'r') as file:
            print("Veryifying the generated phenotypes...")
            genotype = file['0'][:]
            n_indiv, n_snps, _ = genotype.shape
            assert n_indiv == syn_n_subjects
            assert n_snps == inform_snps + noise_snps
            # Checks that at most 3 unique allels exist
            check_genotype_unique_allels(genotype)

    def test_synthetic_phenotypes_generation(self, rep):
       
        labels = generate_syn_phenotypes(ttbr=ttbr,
                                         root_path=SYN_DATA_DIR, n_info_snps=inform_snps, n_noise_snps=noise_snps, quantity=rep)['0']
        assert(labels.shape[0] == syn_n_subjects)

    
    def test_phenotype_invariance(self, rep):
        labels = generate_syn_phenotypes(root_path=SYN_DATA_DIR, quantity=rep)
        
        labels2 = generate_syn_phenotypes(root_path=SYN_DATA_DIR, quantity=rep)
        
    
        for i in range(rep):
            assert np.allclose(labels[str(i)], labels2[str(i)])

    def test_proportion_of_labels(self, rep):
        
        labels = generate_syn_phenotypes(ttbr=ttbr,
                                         root_path=SYN_DATA_DIR, n_info_snps=inform_snps, n_noise_snps=noise_snps, quantity=rep)['0']
        print(sum(labels == 1))
        print(sum(labels == -1))

    def test_feature_map_generation(self):
        """
        From synthetic data in h5py format generate a corresponding feature matrix
        :return: a written featmat at data/synthetic/2d_syn_fm.h5py
        and a written featmat at data/synthetic/3d_syn_fm.h5py
        """
        fm2 = h5py_to_featmat(embedding_type='2d', overwrite=True)
        fm3 = h5py_to_featmat(embedding_type='3d', overwrite=True)
        assert fm2['0'].shape[0] == syn_n_subjects
        assert fm3['0'].shape[0] == syn_n_subjects
        assert fm2['0'].shape[1] == n_total_snps * 3
        assert fm3['0'].shape[1] == n_total_snps


    def test_real_pvalues_generation(self, real_genomic_data, real_labels):
        """
        Computes and save in a .mat file raw P-Values for each disease, chromosome
        :return:
        """
        for disease_id in tqdm(disease_IDs):
            for chrom in tqdm(range(1,23)):
                data = real_genomic_data(disease_id, chrom)
                labels = real_labels(disease_id)

                pvalues = chi_square(data, labels)
                os.makedirs(os.path.dirname(os.path.join(FINAL_RESULTS_DIR,'pvalues',disease_id, str(chrom))), exist_ok=True)
                
                np.save(os.path.join(FINAL_RESULTS_DIR,'pvalues',disease_id, str(chrom)), pvalues)

                del data, pvalues

    def test_mat_preprocessing(self):
        """
        Preprocesses the raw genomic .mat files to remove their trailing 0s
        reshapes from (n_subjects, 3 * n_snps) to (n_subjects, n_snps, 2)
        :return:
        """

        for disease in tqdm(disease_IDs):
            for i in tqdm(range(1, 23)):
                with h5py.File(os.path.join(SYN_DATA_DIR, disease, 'chromo_{}.mat'.format(i)), 'r') as f:
                    chrom = np.array(f.get('X')).T
                    assert chrom.shape[1] > chrom.shape[0]


                    scipy.io.savemat(os.path.join(SYN_DATA_DIR, disease, 'chromo_{}_processed.mat'.format(i)),
                                     {'X': chrom.reshape(chrom.shape[0], -1, 3)[:, :, :2]},
                                     do_compression=True,
                                     appendmat=False)
                    del chrom



