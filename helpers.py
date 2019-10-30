import os

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import chi2
from tensorflow.python.client import device_lib
from tqdm import tqdm

from parameters_complete import SYN_DATA_DIR, ttbr as ttbr, syn_n_subjects, pnorm_feature_scaling, inform_snps, seed, \
    REAL_DATA_DIR


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


def generate_name_from_params(params):
    RUN_DIR_NAME = ""
    for key, val in params.items():
        if key not in ['feature_matrix_path', 'y_path', 'epochs','verbose', 'optimizer', 'batch_size']:
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


def generate_syn_genotypes(root_path=SYN_DATA_DIR, n_subjects=syn_n_subjects, n_info_snps=20, n_noise_snps=10000, quantity=1):
    """ Generate synthetic genotypes and labels by removing all minor allels with low frequency, and missing SNPs.
        First step of data preprocessing, has to be followed by string_to_featmat()
        > Checks that that each SNP in each chromosome has at most 2 unique values in the whole dataset.
    """
    print("Starting synthetic genotypes generation...")

    try:
        os.remove(os.path.join(root_path, 'genomic.h5py'))
    except FileNotFoundError:
        pass


    with h5py.File(os.path.join(REAL_DATA_DIR,'CD','chromo_2.mat'), 'r') as f2:
        chrom2_full = np.array(f2.get('X')).T

    chrom2_full = chrom2_full.reshape(chrom2_full.shape[0], -1, 3)[:, :, :2]
    chrom2_full = remove_small_frequencies(chrom2_full)
    chrom2_full = chrom2_full[:, :n_noise_snps]
    assert chrom2_full.shape[0] > n_subjects
    chrom2 = chrom2_full[:n_subjects]

    with h5py.File(os.path.join(REAL_DATA_DIR,'CD', 'chromo_1.mat'), 'r') as f:
        chrom1_full = np.array(f.get('X')).T


    chrom1_full = chrom1_full.reshape(chrom1_full.shape[0], -1, 3)[:, :, :2]
    chrom1_full = remove_small_frequencies(chrom1_full)
    assert chrom1_full.shape[0] > n_subjects 
    chrom1 = chrom1_full[:n_subjects]  # Keep only 300 people

    # Create #rep different datasets, each with a different set of informative SNPs
    half_noise_size = int(n_noise_snps/2)
    with h5py.File(os.path.join(root_path, 'genomic.h5py'), 'w') as file:

        for i in tqdm(range(quantity)):    

            chrom1_subset = chrom1[:, i*n_info_snps:(i+1)*n_info_snps]

            data = np.concatenate((chrom2[:, :half_noise_size], chrom1_subset,
                                chrom2[:, half_noise_size:half_noise_size*2]), axis=1)
            # If the number of encoded SNPs is insufficient
            if data.shape[1] != n_info_snps + n_noise_snps:
                raise Exception("Not enough SNPs")

            # Write everything!
            file.create_dataset(str(i), data=data)

    return os.path.join(root_path, 'genomic.h5py')


def generate_syn_phenotypes(root_path=SYN_DATA_DIR, tower_to_base_ratio=ttbr, n_info_snps=20, n_noise_snps=10000, quantity=1):
    """
    > Assumes that each SNP has at most 3 unique values in the whole dataset (Two allels and possibly unmapped values)
    IMPORTANT: DOES NOT LOAD FROM FILE
    returns: dict(key, labels)
    """
    print("Starting synthetic phenotypes generation...")

    # Generate Labels from UNIQUE SNP at position 9
    info_snp_idx = int(n_noise_snps/2) + int(n_info_snps/2)

    labels_dict = {}
    def f(genotype, key):
        n_indiv = genotype.shape[0]
        info_snp = genotype[:,  info_snp_idx]  # (n,2)
        major_allel = np.max(info_snp)

        major_mask_1 = (info_snp[:, 0] == major_allel)  # n
        major_mask_2 = (info_snp[:, 1] == major_allel)  # n
        invalid_mask = (info_snp[:, 0] == 48) | (info_snp[:, 1] == 48)

        nb_major_allels = np.sum(
            [major_mask_1, major_mask_2, invalid_mask], axis=0) - 1.0  # n
        probabilities = 1 / \
            (1 + np.exp(-tower_to_base_ratio * (nb_major_allels - np.median(nb_major_allels))))
        random_vector = np.random.RandomState(seed).uniform(low=0.0, high=1.0, size=n_indiv)
        labels = np.where(probabilities > random_vector, 1, -1)
        assert genotype.shape[0] == labels.shape[0]
        labels_dict[key] = labels   
        del genotype


    with h5py.File(os.path.join(root_path, 'genomic.h5py'), 'r') as h5py_file:
        
        Parallel(n_jobs=-1, require='sharedmem')(delayed(f)(h5py_file[str(i)][:],str(i)) for i in tqdm(range(quantity)))
       
    
    return labels_dict





def char_matrix_to_featmat(data, embedding_type='2d', norm_feature_scaling=pnorm_feature_scaling):
    
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
    f_m -= np.mean(f_m, dtype=np.float64, axis=0) # centering
    stddev = ((np.abs(f_m)**norm_feature_scaling).mean(axis=0) * f_m.shape[1])**(1.0/norm_feature_scaling)
    
    # Safe division
    f_m = np.divide(f_m, stddev, out=np.zeros_like(f_m), where=stddev!=0)

    # Reshape Feature matrix
    if embedding_type == '2d':
        pass
    elif embedding_type == '3d':
        f_m = np.reshape(f_m, (n_subjects, num_snp3, 3))

    return f_m.astype(float)





def genomic_to_featmat(embedding_type='2d', overwrite=False):
    """
    Transforms a h5py dictionary of genomic matrix of chars to a tensor of features
    {'0': genomic_mat_0, '1': genomic_mat_1,..., 'rep': genomic_mat_rep }
        =>  {'0': feat_mat_0, '1': feat_mat_1,..., 'rep': feat_mat_rep }
    :param embedding_type: 3d ( n_subjects, 3*n_snps) or 2d ( n_subjects, n_snps, 3)
    :param overwrite:
    :return:
    """

    data_path = os.path.join(SYN_DATA_DIR, 'genomic.h5py')

    fm_path = os.path.join(SYN_DATA_DIR, embedding_type + '_fm.h5py')

    if overwrite:
        try:
            os.remove(fm_path)
        except (FileNotFoundError, OSError):
            pass

    if not overwrite:
        try:
            return h5py.File(fm_path, 'r')
        except (FileNotFoundError, OSError):
            print('Featmat not found: generating new one...')
    
    with h5py.File(fm_path, 'w') as feature_file:
        with h5py.File(data_path, 'r') as data_file:
            for key in tqdm(list(data_file.keys())):
                data = data_file[key][:]
                
                f_m = char_matrix_to_featmat(data, embedding_type)

                feature_file.create_dataset(key, data=f_m)
                del data

    return h5py.File(fm_path, 'r')



def moving_average(weights, window, pnorm_filter):
    """
    Weights postprocessing filter based on a moving average
    Inspired from https://uk.mathworks.com/matlabcentral/fileexchange/12276-moving_average-v3-1-mar-2008
    """


    wnew = weights ** pnorm_filter
    wnew = np.concatenate((
        np.zeros(int((window - 1) / 2 + 1)),
        wnew,
        np.zeros(int((window - 1) / 2))),
        axis=None)
    wnew = np.cumsum(wnew)
    wnew = (wnew[window:] - wnew[0:-window]) ** (1.0 / pnorm_filter)
    wnew /= window ** (1.0 / pnorm_filter)
    return wnew


def chi_square(data, labels):
    """
    Computes pvalues given data and labels
    data: Char matrix (n * n_snp * 2)
    labels: -1, 1 encoding
    filter_indices: array of indices ordering the supposed top k p values
    """

    n_subjects, n_snps, _ = data.shape

    valid_mask = (data[:, :, 0] != 48) & (data[:, :, 1] != 48)  # n , n_snps

    # Find greatest and lowest char code, SNP-wise

    data[data == 48] = 255

    lex_min_per_snp = data.min(axis=(0, 2))  # n_snps

    data[data == 255] = 48

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

def plot_pvalues(complete_pvalues, top_indices_sorted, axes ):
        print("Performing complete X2 to prepare plotting...")
        axes.scatter(range(len(complete_pvalues)),-np.log10(complete_pvalues), marker='.',color='b')
        axes.scatter(top_indices_sorted,-np.log10(complete_pvalues[top_indices_sorted]), marker='x',color='r')
        axes.set_ylabel('-log10(pvalue)')
        axes.set_xlabel('SNP position')

    
def compute_metrics(scores, truth, threshold):
    """
    Computes True Positive Rate, Expected Number of False Positives, Family-wide error rate and precision
    :param scores: Pvalues
    :param truth: Vector of size (n_snps) with 1s where the loci are informative, else 0s
    :param threshold: Loci with scores greater than this value will be considered informative
    :return: tpr, enfr, fwer, precision
    """
    n_experiments = scores.shape[0]
    selected_pvalues_mask = (scores <= threshold) # n, n_snp

    tp = (selected_pvalues_mask & truth).sum()
    fp = (selected_pvalues_mask & np.logical_not(truth)).sum()
    tpr = tp / (n_experiments * inform_snps)
    enfr = fp / n_experiments # ENFR: false rejection of null hyp, that is FALSE POSITIVE
    fwer = ((selected_pvalues_mask & np.logical_not(truth)).sum(axis=1) > 0).sum()/n_experiments
    precision = (tp / (tp + fp)) if (tp + fp)!=0 else 0

    assert 0 <= tpr <= 1
    assert 0 <= fwer <= 1
    assert 0 <= precision <= 1
    return tpr, enfr, fwer, precision

def postprocess_weights(weights,top_k, filter_window_size, p_svm, p_pnorm_filter):
    """
    Rescales weights to unit p-norm, then run a moving average and returns the top_k greatest indices
    :param weights: SVM weights or relevance mapping from which draw the top_k greatest indices
    :param top_k: number of selected indices
    :param filter_window_size: size of window during the moving average
    :param p_svm: norm used in the p_normalization step
    :param p_pnorm_filter: norm used in the moving average
    :return: sorted indices with top k greatest pvalues, averaged weights
    """
    averaged_weights = postprocess_weights_without_avg(weights, p_svm)
    averaged_weights = moving_average(averaged_weights,filter_window_size, p_pnorm_filter)
    top_indices_sorted = averaged_weights.argsort()[::-1][:top_k] # Gets indices of top_k greatest elements
    return top_indices_sorted, averaged_weights

def postprocess_weights_without_avg(weights, p_svm):
    """
    First step of weights postprocessing. Rescales weights to unit 2-norm
    :input  shape (n_snps * 3)
    :return shape (n_snps)
    """
    if np.count_nonzero(weights)==0:
        return np.zeros(weights.reshape(-1, 3).shape[0])
    weights_ = np.absolute(weights)/np.linalg.norm(weights, ord=2)
    weights_ = np.absolute(weights_.reshape(-1, 3))# Group  weights by 3 (yields locus's importance measure)
    weights_ = np.sum(weights_**p_svm, axis=1)**(1.0/p_svm)
    weights_ /= np.linalg.norm(weights_, ord=2)
    return weights_




