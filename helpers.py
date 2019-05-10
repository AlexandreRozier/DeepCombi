from scipy.stats import chi2
import math
import numpy as np
import sklearn.preprocessing as pp
from keras import backend as K
from keras.constraints import Constraint
import os
from tqdm import tqdm
import h5py

from parameters_complete import DATA_DIR

from tqdm import tqdm


def sanitize_chromosome(chromosome):
    # Remove SNPs with unknown values
    allel1_mat = chromosome[:, :, 0]
    allel2_mat = chromosome[:, :, 1]
    valid_mask_1 = np.where(allel1_mat != 48, True, False)
    valid_mask_2 = np.where(allel2_mat != 48, True, False)
    valid_snps = np.logical_and(valid_mask_1, valid_mask_2)  # Elements
    valid_columns = valid_snps.all(axis=0)

    chrom_valid = chromosome[:, valid_columns]  # (n, n_valid_snp, 2)
    n_indiv, _, _ = chrom_valid.shape

    # Remove SNPs with insufficient minor frequencies
    lex_min_per_snp = chrom_valid.min(axis=(0, 2))  # (n)
    lex_min_map = np.tile(lex_min_per_snp, [n_indiv, 1])

    min_mask1 = np.where(chrom_valid[:, :, 0] == lex_min_map, True, False)
    min_mask2 = np.where(chrom_valid[:, :, 1] == lex_min_map, True, False)

    maf = (np.sum(min_mask1, axis=0) +
           np.sum(min_mask2, axis=0))/(2*n_indiv)
    maf = np.minimum(maf, 1-maf)
    chrom_valid = chrom_valid[:, maf > 0.15, :]
    return chrom_valid


def generate_crohn_mat(root_path=DATA_DIR, c=6, n_replication=20, group_size=300, n_info_snps=20, n_noise_snps=10000):

    try:
        os.remove(os.path.join(root_path, 'syn_data.txt'))
    except FileNotFoundError:
        pass

    try:
        os.remove(os.path.join(root_path, 'syn_labels.txt'))

    except FileNotFoundError:
        pass

    f = h5py.File(os.path.join(DATA_DIR, 'chromo_01.mat'), 'r')
    chrom1_full = f.get('X')
    chrom1_full = np.array(chrom1_full).T
    f.close()

    f2 = h5py.File(os.path.join(DATA_DIR, 'chromo_02.mat'), 'r')
    chrom2_full = f2.get('X')
    chrom2_full = np.array(chrom2_full).T
    f2.close()

    for i in tqdm(range(n_replication)):
        chrom1_full = np.roll(chrom1_full, group_size)
        chrom2_full = np.roll(chrom2_full, group_size)

        chrom1 = chrom1_full[:group_size]
        n_indiv, _ = chrom1.shape
        chrom1 = chrom1.reshape(n_indiv, -1, 3)[:, :, :2]
        chrom1 = sanitize_chromosome(chrom1)

        chrom2 = chrom2_full[:group_size]
        n_indiv, _ = chrom2.shape
        chrom2 = chrom2.reshape(n_indiv, -1, 3)[:, :, :2]
        chrom2 = sanitize_chromosome(chrom2)

        half_noise_size = int(n_noise_snps/2)
        data = np.concatenate((chrom2[:, :half_noise_size], chrom1[:, :n_info_snps],
                               chrom2[:, half_noise_size:half_noise_size*2]), axis=1)
        # If the number of encoded SNPs is insufficient
        if data.shape[1] != n_info_snps + n_noise_snps:
            print("SKIPPING RUN")
            pass
        else:
            # Generate Informative  SNPs file
            with open(os.path.join(root_path, 'syn_data.txt'), 'a') as file:
                for i in tqdm(range(n_indiv)):
                    for j in range(n_info_snps+n_noise_snps):
                        all1 = data[i, j, 0]
                        all2 = data[i, j, 1]
                        if all1 < all2:
                            file.write("{}{} ".format(
                                str(chr(all1)), str(chr(all2))))
                        else:
                            file.write("{}{} ".format(
                                str(chr(all2)), str(chr(all1))))
                    file.write("\n")

            # Generate Labels from UNIQUE SNP at position 9

            info_snp_idx = 9
            info_snp = chrom1[:, info_snp_idx]  # (n,2)
            lex_maj = info_snp.max()  # (n,1)
            mask1 = np.where(info_snp[:, 0] == lex_maj, True, False)
            mask2 = np.where(info_snp[:, 1] == lex_maj, True, False)
            nb_major_allels = np.sum([mask1, mask2], axis=0)  # (n,1)
            probabilities = np.power(
                (1+np.exp(-c * (nb_major_allels - np.median(nb_major_allels)))), -1)
            random_vector = np.random.uniform(size=n_indiv)
            labels = np.where(probabilities > random_vector, "1", "-1")

            with open(os.path.join(root_path, 'syn_labels.txt'), 'a') as file:
                for label in tqdm(labels):
                    file.write(label+"\n")
    return os.path.join(root_path, 'syn_data.txt'), os.path.join(root_path, 'syn_labels.txt')


def string_to_featmat(data, data_type_to_be_returned='double', embedding_type='2d'):

    ###  Global Parameters   ###
    (n_subjects, num_snp3) = data.shape

    ##  Main script     ###
    first_strand_by_person_mat = np.vectorize(
        lambda e: bytes(e[0], 'utf-8')[0])(data)
    second_strand_by_person_mat = np.vectorize(
        lambda e: bytes(e[1], 'utf-8')[0])(data)

    num_genotyping_errors = np.sum(first_strand_by_person_mat == b'0'[
                                   0]) + np.sum(second_strand_by_person_mat == b'0'[0])

    # Computes lexicographically highest and lowest nucleotides for each position of each strand
    lexmax_allele_1 = np.max(first_strand_by_person_mat, 0)
    lexmax_allele_2 = np.max(second_strand_by_person_mat, 0)
    lexmax_by_pair = np.maximum(lexmax_allele_1, lexmax_allele_2)

    first_strand_by_person_mat[first_strand_by_person_mat == b'0'[0]] = 255
    second_strand_by_person_mat[second_strand_by_person_mat == b'0'[0]] = 255

    lexmin_allele_1 = np.min(first_strand_by_person_mat, 0)
    lexmin_allele_2 = np.min(second_strand_by_person_mat, 0)
    lexmin_by_pair = np.minimum(lexmin_allele_1, lexmin_allele_2)

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
    feature_map = pp.scale(feature_map, axis=0)  # preprocess matrix

    if(embedding_type == '2d'):
        pass
    elif(embedding_type == '3d'):
        feature_map = np.reshape(feature_map, (n_subjects, num_snp3, 3))

    return feature_map


def count_lines(filename):
    return sum(1 for line in open(filename))


def count_columns(filename):
    with open(filename) as f:
        first_line = f.readline()
        return math.floor(len(first_line)/3)


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


def chi_square(data, labels, filter_indices=[]):
    """
    data: 2D encoding
    labels: +1, -1 encoding
    filter_indices: array of indices ordering the supposed top k p values
    """

    ##########################
    ###     Assertions     ###
    ##########################
    individuals, n_snps = data.shape

    if(len(filter_indices) > 0):
        data = np.take(data, filter_indices, axis=1)
        assert(data.shape[1] == len(filter_indices))
        individuals, n_snps = data.shape

    ##########################
    ###   Initialization   ###
    ##########################

    invalid_value = b'0'[0]
    chisq_value = np.zeros(n_snps)

    ##########################
    ###    Prepare data    ###
    ##########################

    allele1 = np.vectorize(
        lambda expected: bytes(expected[0], 'utf-8')[0])(data)
    allele2 = np.vectorize(
        lambda expected: bytes(expected[1], 'utf-8')[0])(data)

    ##################################
    ##################################

    invalid_mask = np.logical_or(
        np.where(allele1 == invalid_value, True, False),
        np.where(allele2 == invalid_value, True, False))
    valid_mask = np.logical_not(invalid_mask)
    # [n_snps], for each SNP, nb of individuals bearing a valid sequencing of this SNP
    n_valid_bearers_by_snp = np.sum(valid_mask, 0)

    # Find greatest and lowest char code, SNP-wise

    allele1[allele1 == b'0'[0]] = 255
    allele2[allele2 == b'0'[0]] = 255

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
