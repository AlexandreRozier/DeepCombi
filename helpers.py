from scipy.stats import chi2
import math
import numpy as np
import sklearn.preprocessing as pp
from keras import backend as K
from keras.constraints import Constraint
import os
import io
from tqdm import tqdm
import h5py
import keras
from parameters_complete import DATA_DIR

from tqdm import tqdm

def generate_name_from_params(params):
    RUN_DIR_NAME = ""
    for key, val in params.items():
        if key not in ['feature_matrix_path','y_path', 'verbose']:
            RUN_DIR_NAME+= "{}-{}__".format(key,val)
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
    lex_min = np.tile(np.min(chromosome, axis=(0,2)),[n_indiv,1])
    
    allel1= chromosome[:,:,0]
    allel2= chromosome[:,:,1]
    maf = (np.sum(allel1==lex_min,axis=0) + np.sum(allel2==lex_min, axis=0))/(2*n_indiv)
    maf = np.minimum(maf, 1-maf)
    

    chrom_low_f_removed = chromosome[:, maf > 0.15, :]
    chrom_low_f_removed[chrom_low_f_removed == 255] = 48

    check_genotype_unique_allels(chrom_low_f_removed)
    
    return chrom_low_f_removed

def check_genotype_unique_allels(genotype):
    assert(max([len(np.unique(genotype[:,i,:])) for i in range(genotype.shape[1])]) <=3)

def generate_syn_genotypes(root_path=DATA_DIR,prefix="syn", n_subjects=20, n_info_snps=20, n_noise_snps=10000):
    
    """ Generate synthetic genotypes and labels by removing all minor allels with low frequency, and missing SNPs.
        First step of data preprocessing, has to be followed by string_to_featmat()
        > Checks that that each SNP in each chromosome has at most 2 unique values in the whole dataset.
    
    """
    print("Starting synthetic genotypes generation...")

    try:
        os.remove(os.path.join(root_path, prefix+'_data.h5py'))
    except FileNotFoundError:
        pass

    # Load complete chromosomes
    with h5py.File(os.path.join(DATA_DIR, 'chromo_01.mat'), 'r') as f:
        chrom1_full = np.array(f.get('X')).T

    chrom1_full = chrom1_full.reshape(chrom1_full.shape[0], -1, 3)[:, :, :2]
    chrom1_full = remove_small_frequencies(chrom1_full)
    chrom1_full = chrom1_full[:,:n_info_snps]

    with h5py.File(os.path.join(DATA_DIR, 'chromo_02.mat'), 'r') as f2:
        chrom2_full = np.array(f2.get('X')).T

    chrom2_full = chrom2_full.reshape(chrom2_full.shape[0], -1, 3)[:, :, :2]
    chrom2_full = remove_small_frequencies(chrom2_full) 
    chrom2_full = chrom2_full[:,:n_noise_snps]

    half_noise_size = int(n_noise_snps/2)

    assert(n_subjects < chrom1_full.shape[0] and n_subjects < chrom2_full.shape[0])

    chrom1 = chrom1_full[:n_subjects] # Keep only 300 people
    chrom2 = chrom2_full[:n_subjects]

    data = np.concatenate((chrom2[:, :half_noise_size], chrom1,
                            chrom2[:, half_noise_size:half_noise_size*2]), axis=1)
    # If the number of encoded SNPs is insufficient
    if data.shape[1] != n_info_snps + n_noise_snps:
        raise Exception("Not enough SNPs")
        

    # Write everything!
    with h5py.File(os.path.join(root_path, prefix+'_data.h5py'), 'w') as file:
        file.create_dataset("X", data=data)

    return os.path.join(root_path, prefix+'_data.h5py')
    
def generate_syn_phenotypes(root_path=DATA_DIR, prefix="syn", c=6, n_info_snps=20, n_noise_snps=10000):
    """
    > Assumes that each SNP has at most 3 unique values in the whole dataset (Two allels and possibly unmapped values)
    """
    print("Starting synthetic phenotypes generation...")

    try:
        os.remove(os.path.join(root_path, prefix+'_labels.h5py'))
    except FileNotFoundError:
        pass

    # Generate Labels from UNIQUE SNP at position 9
    info_snp_idx = int(n_noise_snps/2) + int(n_info_snps/2)

    with h5py.File(os.path.join(root_path, prefix+'_data.h5py'), 'r') as file:
       
        genotype = file['X'][:]

    n_indiv = genotype.shape[0]
    info_snp = genotype[:,  info_snp_idx]  # (n,2)
    all1 = np.max(info_snp)
    info_snp[info_snp==48]= 255
    all2 = np.min(info_snp)
    minor_allel = all1 if np.sum(np.where(info_snp==all1,True,False)) < np.sum(np.where(info_snp==all2,True,False)) else all2
    major_allel = all1 if np.sum(np.where(info_snp==all1,True,False)) > np.sum(np.where(info_snp==all2,True,False)) else all2
    assert(minor_allel != major_allel)

    major_mask_1 = np.where(info_snp[:, 0] == major_allel, True, False)  # n
    major_mask_2 = np.where(info_snp[:, 1] == major_allel, True, False) # n

    nb_major_allels = np.sum([major_mask_1, major_mask_2], axis=0)  # n
    probabilities = (1+np.exp(-c * (nb_major_allels - np.median(nb_major_allels)))) ** -1
    random_vector = np.random.uniform(low=0.0, high=1.0, size=n_indiv)
    labels = np.where(probabilities > random_vector, 1, -1)
    assert genotype.shape[0] == labels.shape[0]
    
    with h5py.File(os.path.join(root_path, prefix+'_labels.h5py'), 'w') as file:
        file.create_dataset("X", data=labels)


    return os.path.join(root_path, prefix+'_labels.h5py')



    
def string_to_featmat(data, embedding_type='2d', overwrite=False):
    """ - Transforms numpy matrix of chars to a tensor of features.
        - Encode SNPs with error to [0,0,0] 
        - Does NOT change shape or perform modifications
        Second and final step of data preprocessing.
    """

    filename = os.path.join(DATA_DIR,embedding_type +'_feature_matrix.h5py')

    # Check if feature matrix has already been generated
    if not overwrite:
        try:
            print("Loading feature matrix from disk")
            with h5py.File(filename, 'r') as file:

                return file['X'][:]

        except (FileNotFoundError, OSError) as e:
            print("Feature matrix at {} not found, generating new one".format(filename))

    ###  Global Parameters   ###
    (n_subjects, num_snp3, _) = data.shape

    first_strand_by_person_mat = data[:,:,0]
    second_strand_by_person_mat = data[:,:,1]


    # Computes lexicographically highest and lowest nucleotides for each position of each strand
    lexmax_by_pair = np.max(data, axis=(0,2))
    data[data==48] = 255

    lexmin_by_pair = np.min(data, axis=(0,2))

    # Masks showing valid or invalid indices
    # SNPs being unchanged amongst the whole dataset, hold no information
    invalid_bool_mask = np.tile(
        np.where(lexmin_by_pair == lexmax_by_pair, True, False), [n_subjects, 1])
    

    lexmin_mask = np.tile(lexmin_by_pair, [n_subjects, 1])
    lexmax_mask = np.tile(lexmax_by_pair, [n_subjects, 1])

    # indices where allel1 equals the lowest value
    allele1_lexminor_bool_mask = np.where(
        first_strand_by_person_mat == lexmin_mask, True, False)
    # indices where allel1 equals the lowest value
    allele1_lexmajor_bool_mask = np.where(
        first_strand_by_person_mat == lexmax_mask, True, False)
    # indices where allel1 equals the lowest value
    allele2_lexminor_bool_mask = np.where(
        second_strand_by_person_mat == lexmin_mask, True, False)
    # indices where allel1 equals the lowest value
    allele2_lexmajor_bool_mask = np.where(
        second_strand_by_person_mat == lexmax_mask, True, False)

    first_strand_by_person_mat[allele1_lexmajor_bool_mask] = 2
    feature_map = np.zeros((n_subjects, num_snp3), dtype=(int, 3))

    feature_map[np.logical_and(allele1_lexminor_bool_mask,
                               allele2_lexminor_bool_mask)] = [1, 0, 0]
    feature_map[np.logical_or(
        np.logical_and(allele1_lexmajor_bool_mask,
                       allele2_lexminor_bool_mask),
        np.logical_and(allele1_lexminor_bool_mask, allele2_lexmajor_bool_mask))
    ] = [0, 1, 0]
    feature_map[np.logical_and(allele1_lexmajor_bool_mask,
                               allele2_lexmajor_bool_mask)] = [0, 0, 1]
    feature_map[invalid_bool_mask] = [0, 0, 0]
    feature_map = np.reshape(feature_map, (n_subjects, 3*num_snp3))
    feature_map = feature_map.astype(np.double)

    
    # Reshape Feature matrix
    if(embedding_type == '2d'):
        pass
    elif(embedding_type == '3d'):
        feature_map = np.reshape(feature_map, (n_subjects, num_snp3, 3))
    
    # Write!
    with h5py.File(filename, 'w') as file:
        file.create_dataset("X", data=feature_map)


    return feature_map


def count_lines(filename):
    #return sum(1 for line in open(filename))
    with h5py.File(filename,'r') as d:
        return d['X'].shape[0]


def moving_average(w, k, power=1):
    """
    Inspired from https://uk.mathworks.com/matlabcentral/fileexchange/12276-moving_average-v3-1-mar-2008
    """
    assert(k % 2 == 1)

    wnew = np.absolute(w)
    wnew = np.power(wnew, power)
    wnew = np.concatenate((
        np.zeros(int((k-1)/2+1)),
        wnew,
        np.zeros(int((k-1)/2))),
        axis=None)
    wnew = np.cumsum(wnew)
    assert(wnew[k:].shape == wnew[0:-k].shape)
    wnew = np.subtract(wnew[k:], wnew[0:-k])
    wnew = np.power(wnew, 1.0/power)
    wnew = np.divide(wnew, k**(1.0/power))
    return wnew

# TODO test this


def simple_moving_average(w, k):
    return np.convolve(w, np.ones(k), 'same')

def other_moving_avg(x, k, p):
    x = np.absolute(x)
    x = np.power(x, p)
    d = len(x)
    result = np.zeros(d)
    for j in range(d):
        acc = 0
        for l in range(max(0, j-int((k-1)/2)), min(d, j+int((k-1)/2))):
            acc += x[l]
        result[j] = acc
    return np.power(result, 1.0/p)


def chi_square(data, labels):
    """
    data: Char matrix (n * n_snp * 2)
    labels: -1, 1 encoding
    filter_indices: array of indices ordering the supposed top k p values
    """

   
    individuals, n_snps, _ = data.shape
   
    invalid_value = 48
    chisq_value = np.zeros(n_snps)

    allele1 = data[:,:,0]
    allele2 = data[:,:,1]

    ##################################
    ##################################

    invalid_mask = np.logical_or(
        np.where(allele1 == invalid_value, True, False),
        np.where(allele2 == invalid_value, True, False))
    valid_mask = np.logical_not(invalid_mask)
    # [n_snps], for each SNP, nb of individuals bearing a valid sequencing of this SNP
    n_valid_bearers_by_snp = np.sum(valid_mask, 0)

    # Find greatest and lowest char code, SNP-wise

    allele1[allele1 == 48] = 255
    allele2[allele2 == 48] = 255

    lex_min = np.minimum(
        np.min(allele1, 0),
        np.min(allele2, 0))

    # Valid case masks
    cases_mask = np.where(labels == 1, True, False)
    cases_mask = np.repeat(cases_mask, n_snps, axis=0).reshape((-1, n_snps))
    valid_cases_mask = np.logical_and(valid_mask, cases_mask)
    num_valid_cases = np.sum(valid_cases_mask, axis=0)

    # Valid control masks
    controls_mask = np.where(labels == -1, True, False)
    controls_mask = np.repeat(controls_mask, n_snps,
                              axis=0).reshape((-1, n_snps))
    valid_controls_mask = np.logical_and(valid_mask, controls_mask)
    num_valid_controls = np.sum(valid_controls_mask, axis=0)

    # Allel pairs masks
    lex_min_mat = np.tile(lex_min, [individuals, 1])
    minor_allel1_mask = np.where(allele1 == lex_min_mat, True, False)
    minor_allel2_mask = np.where(allele2 == lex_min_mat, True, False)
    minor_allels_mask = np.logical_and.reduce(
        (minor_allel1_mask, minor_allel2_mask, valid_mask))
    both_allels_mask = np.where(allele1 != allele2, True, False)
    both_allels_mask = np.logical_and(both_allels_mask,  valid_mask)

    # Populate valid_table
    """
    This table is a per-SNP lookup table for the allel combinations (rows) and case or control nature (columns)
    valid_table[0] : cases
    valid_table[1] : controls
    valid_table[i,j]: [n_snp], for each SNP genotype, number of individuals bearing it
    """
    valid_table = np.zeros((2, 3, n_snps))
    valid_table[0, 0] = np.sum(np.logical_and.reduce(
        (minor_allels_mask, valid_cases_mask)), axis=0)  # 2 Minor allels, case
    valid_table[0, 1] = np.sum(np.logical_and(
        both_allels_mask, valid_cases_mask), axis=0)  # Both allels, case
    valid_table[0, 2] = num_valid_cases - valid_table[0, 0] - \
        valid_table[0, 1]  # 2 Major allels, case

    valid_table[1, 0] = np.sum(np.logical_and.reduce(
        (minor_allels_mask, valid_controls_mask)), axis=0)  # 2 Minor allels, control
    valid_table[1, 1] = np.sum(np.logical_and(
        both_allels_mask, valid_controls_mask), axis=0)  # Both allels, control
    valid_table[1, 2] = num_valid_controls - valid_table[1, 0] - \
        valid_table[1, 1]  # 2 Major allels, case

    """
    # compute minor allele frequencies per SNP (MAFs)
    n_ma_per_snp = 2*(valid_table[0,0] + valid_table[1,0]) + valid_table[0,1] + valid_table[1,1]
    ma_freq_per_snp = np.divide(n_ma_per_snp, 2*n_valid_bearers_by_snp)
    assert(np.max(ma_freq_per_snp) <= 1 and np.min(ma_freq_per_snp >=0))
    ma_freq_per_snp = np.minimum(ma_freq_per_snp, (1.0 - ma_freq_per_snp)) 
    """

    # compute p-values
    # ~ nb of cases, nb of controls
    case_ctrl_table = np.squeeze(np.sum(valid_table, axis=1))
    # ~ nb of minor , mixed and major pairs
    genotype_table = np.squeeze(np.sum(valid_table, axis=0))

    valid_n_by_snp = np.sum(case_ctrl_table, axis=0)
    chisq_value = np.zeros(n_snps)

    for i in range(2):
        for j in range(3):
            # Independence assumption: e = nb_c_or_c * genotype_table / N
            expected = np.divide(np.multiply(
                case_ctrl_table[i], genotype_table[j]), valid_n_by_snp)
            # Expected can be 0 if dataset does not contain this (genotype, group) combination
            nonzero_mask = np.where(expected != 0, True, False)
            observed = valid_table[i, j, nonzero_mask]
            zero_mask = np.logical_not(nonzero_mask)
            if (np.sum(zero_mask) != len(expected)):  # If expected has at least one nonzero value

                chisq_value[nonzero_mask] = chisq_value[nonzero_mask] + np.divide(
                    (observed - expected[nonzero_mask])**2, expected[nonzero_mask])

    pvalues = chi2.sf(chisq_value, 2)
    return pvalues



class EnforceNeg(Constraint):
    """Constrains the weights to be negative.
    """

    def __call__(self, w):
        w *= K.cast(K.greater_equal(-w, 0.), K.floatx())
        return w
