
import numpy as np


def string_to_featmat(data, data_type_to_be_returned='double', feature_embedding='genotypic'):

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
    invalid_bool_mask = np.tile(np.where(lexmin_by_pair == lexmax_by_pair, True, False),[n_subjects, 1])

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
    feature_map = None
    if(feature_embedding == 'genotypic'):
        feature_map = np.zeros((n_subjects, num_snp3),dtype=(int,3))

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
    return feature_map
       
    """
      case 'allelic'
        featmat = false(4*num_snp, n_subjects)
        featmat(1: 4: end, : ) = allele1_lexminor
        featmat(2: 4: end, : ) = allele1_lexmajor
        featmat(3: 4: end, : ) = allele2_lexminor
        featmat(4: 4: end, : ) = allele2_lexmajor
        featmat((invalid-1)*4+1, : ) = 0
        featmat((invalid-1)*4+2, : ) = 0
        featmat((invalid-1)*4+3, : ) = 0
        featmat((invalid-1)*4+4, : ) = 0
      case 'nominal'
        marginals = [sum(allele1_lexminor, 2) + sum(allele2_lexminor, 2),
                      sum(allele2_lexmajor, 2) + sum(allele1_lexmajor, 2)]
        [foo risk_allele] = min(marginals, [], 2)
        risk_allele_is_lexminor = (risk_allele == 1)
        risk_allele_is_lexmajor = (risk_allele == 2)
        featmat = (allele1_lexminor + allele2_lexminor) .* repmat(risk_allele_is_lexminor, [1, n_subjects]) ...
                      + (allele1_lexmajor + allele2_lexmajor) .* repmat(risk_allele_is_lexmajor, [1, n_subjects])
        featmat(invalid, : ) = 0
      otherwise
        error('type of feature embedding invalid')
    end


#################################
##  Data type to be returned  ###
#################################

switch data_type_to_be_returned
  case 'single'
    featmat = single(featmat)
  case 'double'
    featmat = double(featmat)
  case 'logical'
    featmat = logical(featmat)
  case 'uint8'
    featmat = uint8(featmat)
end
"""
