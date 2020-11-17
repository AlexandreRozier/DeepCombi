import os

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import chi2
from tensorflow.python.client import device_lib
from tqdm import tqdm


def char_matrix_to_featmat_no_scaling(data, embedding_type='2d', norm_feature_scaling=6):
    
    ###  Global Parameters   ###
    (n_subjects, num_snp3, _) = data.shape

    # Computes lexicographically highest and lowest nucleotides for each position of each strand
    lexmax_overall_per_snp = np.max(data, axis=(0, 2))
    data[data == 48] = 255

    lexmin_overall_per_snp = np.min(data, axis=(0, 2))
    data[data == 255] = 48

    # Masks showing valid or invalid indices
    # SNPs being unchanged amongst the whole dataset, hold no information

    lexmin_mask_per_snp = np.tile(lexmin_overall_per_snp, [n_subjects, 1])
    lexmax_mask_per_snp = np.tile(lexmax_overall_per_snp, [n_subjects, 1])

    invalid_bool_mask = (lexmin_mask_per_snp == lexmax_mask_per_snp)

    allele1 = data[:, :, 0]
    allele2 = data[:, :, 1]

    # indices where allel1 equals the lowest value
    allele1_lexminor_mask = (allele1 == lexmin_mask_per_snp)
    # indices where allel1 equals the highest value
    allele1_lexmajor_mask = (allele1 == lexmax_mask_per_snp)
    # indices where allel2 equals the lowest value
    allele2_lexminor_mask = (allele2 == lexmin_mask_per_snp)
    # indices where allel2 equals the highest value
    allele2_lexmajor_mask = (allele2 == lexmax_mask_per_snp)

    f_m = np.zeros((n_subjects, num_snp3), dtype=(int, 3))

    f_m[allele1_lexminor_mask & allele2_lexminor_mask] = [1, 0, 0]
    f_m[(allele1_lexmajor_mask & allele2_lexminor_mask) |
        (allele1_lexminor_mask & allele2_lexmajor_mask)] = [0, 1, 0]
    f_m[allele1_lexmajor_mask & allele2_lexmajor_mask] = [0, 0, 1]
    f_m[invalid_bool_mask] = [0, 0, 0]
    f_m = np.reshape(f_m, (n_subjects, 3*num_snp3))
    f_m = f_m.astype(np.double)

    ## Rescale feature matrix
    #f_m -= np.mean(f_m, dtype=np.float64, axis=0) # centering
    #stddev = ((np.abs(f_m)**norm_feature_scaling).mean(axis=0) * f_m.shape[1])**(1.0/norm_feature_scaling)
    
    ## Safe division
    #f_m = np.divide(f_m, stddev, out=np.zeros_like(f_m), where=stddev!=0)

    # Reshape Feature matrix
    if embedding_type == '2d':
        pass
    elif embedding_type == '3d':
        f_m = np.reshape(f_m, (n_subjects, num_snp3, 3))

    return f_m.astype(float)



