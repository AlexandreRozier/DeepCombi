from scipy.stats import chi2
import math
import numpy as np
import sklearn.preprocessing as pp
import tensorflow as tensorflow
from keras import backend as K
from keras.constraints import Constraint
import os
import io
from tqdm import tqdm
import h5py
import keras
from tqdm import tqdm
from parameters_complete import DATA_DIR, ttbr as ttbr, n_subjects, pnorm_feature_scaling
from parameters_complete import rep

from tqdm import tqdm


def generate_name_from_params(params):
    RUN_DIR_NAME = ""
    for key, val in params.items():
        if key not in ['feature_matrix_path', 'y_path', 'verbose']:
            RUN_DIR_NAME += "{}-{}__".format(key, val)
    RUN_DIR_NAME += "{}".format(os.environ['SGE_TASK_ID'])
    return RUN_DIR_NAME


def remove_small_frequencies(chromosome):
    """
    This returns a chromosom with only minor allel freq > 0.15
    This chromosom can be safely used to generate synthetic genotypes.
    The returned value CAN CONTAIN UNMAPPED SNPs !
    """
    chromosome[chromosome == 48] = 255
    n_indiv = chromosome.shape[0]
    lex_min = np.tile(np.min(chromosome, axis=(0, 2)), [n_indiv, 1])

    allel1 = chromosome[:, :, 0]
    allel2 = chromosome[:, :, 1]
    lexmin_mask_1 = (allel1 == lex_min)
    lexmin_mask_2 = (allel2 == lex_min)
    maf = (lexmin_mask_1.sum(0) + lexmin_mask_2.sum(0))/(2*n_indiv)  # n_snps
    maf = np.minimum(maf, 1-maf)
    chromosome[chromosome == 255] = 48

    chrom_low_f_removed = chromosome[:, maf > 0.15, :]
    chrom_low_f_removed.sort()
    check_genotype_unique_allels(chrom_low_f_removed)

    return chrom_low_f_removed


def check_genotype_unique_allels(genotype):
    assert(max([len(np.unique(genotype[:, i, :]))
                for i in range(genotype.shape[1])]) <= 3)


def generate_syn_genotypes(root_path=DATA_DIR, prefix="syn", n_subjects=n_subjects, n_info_snps=20, n_noise_snps=10000, quantity=rep):
    """ Generate synthetic genotypes and labels by removing all minor allels with low frequency, and missing SNPs.
        First step of data preprocessing, has to be followed by string_to_featmat()
        > Checks that that each SNP in each chromosome has at most 2 unique values in the whole dataset.
        # TODO When rep > 1, choose random consecutive informative SNPs
    """
    print("Starting synthetic genotypes generation...")

    try:
        os.remove(os.path.join(root_path, prefix+'_data.h5py'))
    except FileNotFoundError:
        pass


    with h5py.File(os.path.join(DATA_DIR, 'chromo_02.mat'), 'r') as f2:
        chrom2_full = np.array(f2.get('X')).T

    chrom2_full = chrom2_full.reshape(chrom2_full.shape[0], -1, 3)[:, :, :2]
    chrom2_full = remove_small_frequencies(chrom2_full)
    chrom2_full = chrom2_full[:, :n_noise_snps]
    assert chrom2_full.shape[0] > n_subjects
    chrom2 = chrom2_full[:n_subjects]

    with h5py.File(os.path.join(DATA_DIR, 'chromo_01.mat'), 'r') as f:
        chrom1_full = np.array(f.get('X')).T


    chrom1_full = chrom1_full.reshape(chrom1_full.shape[0], -1, 3)[:, :, :2]
    chrom1_full = remove_small_frequencies(chrom1_full)
    assert chrom1_full.shape[0] > n_subjects 
    chrom1 = chrom1_full[:n_subjects]  # Keep only 300 people

    # Create #rep different datasets, each with a different set of informative SNPs
    half_noise_size = int(n_noise_snps/2)
    with h5py.File(os.path.join(root_path, prefix+'_data.h5py'), 'w') as file:

        for i in tqdm(range(quantity)):    

            chrom1_subset = chrom1[:, i*n_info_snps:(i+1)*n_info_snps]

            data = np.concatenate((chrom2[:, :half_noise_size], chrom1_subset,
                                chrom2[:, half_noise_size:half_noise_size*2]), axis=1)
            # If the number of encoded SNPs is insufficient
            if data.shape[1] != n_info_snps + n_noise_snps:
                raise Exception("Not enough SNPs")

            # Write everything!
            file.create_dataset(str(i), data=data)

    return os.path.join(root_path, prefix+'_data.h5py')


def generate_syn_phenotypes(root_path=DATA_DIR, ttbr=ttbr, prefix="syn", n_info_snps=20,  n_noise_snps=10000):
    """
    > Assumes that each SNP has at most 3 unique values in the whole dataset (Two allels and possibly unmapped values)
    IMPORTANT: DOES NOT LOAD FROM FILE
    """
    print("Starting synthetic phenotypes generation...")


    # Generate Labels from UNIQUE SNP at position 9
    info_snp_idx = int(n_noise_snps/2) + int(n_info_snps/2)

    with h5py.File(os.path.join(root_path, prefix+'_data.h5py'), 'r') as file:
        labels_dict = {}
        for key in list(file.keys()):
            genotype = file[key][:]

            n_indiv = genotype.shape[0]
            info_snp = genotype[:,  info_snp_idx]  # (n,2)
            major_allel = np.max(info_snp)

            major_mask_1 = (info_snp[:, 0] == major_allel)  # n
            major_mask_2 = (info_snp[:, 1] == major_allel)  # n
            invalid_mask = (info_snp[:, 0] == 48) | (info_snp[:, 1] == 48)

            nb_major_allels = np.sum(
                [major_mask_1, major_mask_2, invalid_mask], axis=0) - 1.0  # n
            probabilities = 1 / \
                (1+np.exp(-ttbr * (nb_major_allels - np.median(nb_major_allels))))
            random_vector = np.random.uniform(low=0.0, high=1.0, size=n_indiv)
            labels = np.where(probabilities > random_vector, 1, -1)
            assert genotype.shape[0] == labels.shape[0]
            labels_dict[key] = labels
    
    return labels_dict


def string_to_featmat(data, embedding_type='2d'):
    """ - Transforms numpy matrix of chars to a tensor of features.
        - Encode SNPs with error to [0,0,0] 
        - Does NOT change shape or perform modifications
        Second and final step of data preprocessing.
    """
    
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

    # Rescale feature matrix
    f_m -= np.mean(f_m, axis=0)
    stddev = (np.mean(np.abs(f_m)**pnorm_feature_scaling, axis=0)
              * f_m.shape[1])**(1.0/pnorm_feature_scaling)
    
    # Safe division
    f_m = np.divide(f_m, stddev, out=np.zeros_like(f_m), where=stddev!=0)

   
    # Reshape Feature matrix
    if(embedding_type == '2d'):
        pass
    elif(embedding_type == '3d'):
        f_m = np.reshape(f_m, (n_subjects, num_snp3, 3))

    
    return f_m



def moving_average(w, k, p=1):
    """
    Inspired from https://uk.mathworks.com/matlabcentral/fileexchange/12276-moving_average-v3-1-mar-2008
    """

    if k == 1:
        return w

    assert(k % 2 == 1)

    wnew = w ** p
    wnew = np.concatenate((
        np.zeros(int((k-1)/2+1)),
        wnew,
        np.zeros(int((k-1)/2))),
        axis=None)
    wnew = np.cumsum(wnew)
    wnew = (wnew[k:] - wnew[0:-k])**(1.0/p)
    wnew /= k**(1.0/p)
    return wnew


def chi_square(data, labels):
    """
    data: Char matrix (n * n_snp * 2)
    labels: -1, 1 encoding
    filter_indices: array of indices ordering the supposed top k p values
    """

    n_subjects, n_snps, _ = data.shape

    valid_mask = (data[:, :, 0] != 48) & (data[:, :, 1] != 48)  # n , n_snps

    # Find greatest and lowest char code, SNP-wise

    data[data == 48] == 255

    lex_min_per_snp = data.min(axis=(0, 2))  # n_snps

    data[data == 255] == 48

    # Valid case masks - n * n_snp
    cases_mask = (labels == 1)
    cases_mask = np.repeat(cases_mask, n_snps, axis=0).reshape((-1, n_snps))
    controls_mask = (labels == -1)
    controls_mask = np.repeat(controls_mask, n_snps,
                              axis=0).reshape((-1, n_snps))

    # Valid control masks
    valid_cases_mask = valid_mask & cases_mask
    num_valid_cases = np.sum(valid_cases_mask, axis=0)
    valid_controls_mask = valid_mask & controls_mask
    num_valid_controls = np.sum(valid_controls_mask, axis=0)

    # Allel pairs masks
    lex_min_mat = np.tile(lex_min_per_snp, [n_subjects, 1])

    minor_allels_mask = (data[:, :, 0] == lex_min_mat) & (
        data[:, :, 1] == lex_min_mat)
    both_allels_mask = (data[:, :, 0] != data[:, :, 1])

    # Populate valid_table
    """
    This table is a per-SNP lookup table for the allel combinations (rows) and case or control nature (columns)
    valid_table[0] : cases
    valid_table[1] : controls
    valid_table[i,j]: [n_snp], for each SNP genotype, number of individuals bearing it
    """
    valid_table = np.zeros((2, 3, n_snps))
    valid_table[0, 0] = (minor_allels_mask & valid_cases_mask).sum(
        axis=0)  # 2 Minor allels, case
    valid_table[0, 1] = (both_allels_mask & valid_cases_mask).sum(
        axis=0)  # Both allels, case
    valid_table[0, 2] = num_valid_cases - valid_table[0, 0] - \
        valid_table[0, 1]  # 2 Major allels, case

    valid_table[1, 0] = (minor_allels_mask & valid_controls_mask).sum(
        axis=0)  # 2 Minor allels, control
    valid_table[1, 1] = (both_allels_mask & valid_controls_mask).sum(
        axis=0)  # Both allels, control
    valid_table[1, 2] = num_valid_controls - valid_table[1, 0] - \
        valid_table[1, 1]  # 2 Major allels, case

    # compute p-values
    row_marginals = np.squeeze(np.sum(valid_table, axis=1))
    col_marginals = np.squeeze(np.sum(valid_table, axis=0))

    valid_n_by_snp = np.sum(row_marginals, axis=0)
    chisq_value = np.zeros(n_snps)

    for i in range(2):
        for j in range(3):
            # Independence assumption: e = nb_c_or_c * genotype_table / N
            e = row_marginals[i] * col_marginals[j] / valid_n_by_snp
            # Expected can be 0 if dataset does not contain this (genotype, group) combination
            nonzero_mask = (e != 0)
            observed = valid_table[i, j, nonzero_mask]
            zero_mask = np.logical_not(nonzero_mask)
            if (np.sum(zero_mask) != len(e)):  # If expected has at least one nonzero value

                chisq_value[nonzero_mask] += (observed -
                                              e[nonzero_mask])**2 / e[nonzero_mask]

    pvalues = chi2.sf(chisq_value, 2)
    return pvalues


class EnforceNeg(Constraint):
    """Constrains the weights to be negative.
    """

    def __call__(self, w):
        w *= K.cast(K.greater_equal(-w, 0.), K.floatx())
        return w
