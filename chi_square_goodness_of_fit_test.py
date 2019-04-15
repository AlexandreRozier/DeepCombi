import numpy as np
from scipy.stats import chi2

def chi_square_goodness_of_fit_test(data,labels):

    ##########################
    ###     Assertions     ###
    ##########################
    individuals, n_snps = data.shape

    ##########################
    ###   Initialization   ###
    ##########################

    invalid_value = b'0'[0]
    lex_min = np.zeros(n_snps)
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
        np.where(allele1==invalid_value, True, False),
        np.where(allele2==invalid_value, True, False))
    valid_mask = np.logical_not(invalid_mask)
    n_valid_bearers_by_snp=np.sum(valid_mask, 0) # [n_snps], for each SNP, nb of individuals bearing a valid sequencing of this SNP

   
    
    # Find greatest and lowest char code, SNP-wise

    allele1[allele1 == b'0'[0]] = 255
    allele2[allele2 == b'0'[0]] = 255

    lex_min_1 = np.min(allele1, 0)
    lex_min_2 = np.min(allele2, 0)
    lex_min = np.minimum(lex_min_1, lex_min_2)

    # Valid case masks
    cases_mask = np.where(labels==1, True, False)
    cases_mask = np.repeat(cases_mask,n_snps, axis=0).reshape((-1,n_snps))
    valid_cases_mask = np.logical_and(valid_mask ,cases_mask)
    num_valid_cases = np.sum(valid_cases_mask, axis=0)

    # Valid control masks
    controls_mask = np.where(labels==-1, True, False)
    controls_mask = np.repeat(controls_mask, n_snps, axis=0).reshape((-1,n_snps))
    valid_controls_mask = np.logical_and(valid_mask , controls_mask)
    num_valid_controls = np.sum(valid_controls_mask, axis=0)

    # Allel pairs masks
    lex_min_mat = np.tile(lex_min, [individuals,1])
    minor_allel1_mask = np.where(allele1==lex_min_mat, True, False)
    minor_allel2_mask = np.where(allele2==lex_min_mat, True, False)
    minor_allels_mask = np.logical_and.reduce((minor_allel1_mask, minor_allel2_mask, valid_mask))
    both_allels_mask = np.where(allele1!=allele2, True, False)
    both_allels_mask = np.logical_and(both_allels_mask,  valid_mask)

    
    # Populate valid_table
    """
    This table is a per-SNP lookup table for the allel combinations (rows) and case or control nature (columns)
    valid_table[0] : cases
    valid_table[1] : controls
    valid_table[i,j]: [n_snp], for each SNP allel combination, number of individuals bearing it
    """
    valid_table = np.zeros((2,3,n_snps))
    valid_table[0,0] = np.sum(np.logical_and.reduce((minor_allels_mask, valid_cases_mask)),axis=0) # 2 Minor allels, case
    valid_table[0,1] = np.sum(np.logical_and(both_allels_mask,valid_cases_mask), axis=0) # Both allels, case
    valid_table[0,2] = num_valid_cases - valid_table[0,0] - valid_table[0,1] # 2 Major allels, case

    valid_table[1,0] = np.sum(np.logical_and.reduce((minor_allels_mask, valid_controls_mask)), axis=0) # 2 Minor allels, control
    valid_table[1,1] = np.sum(np.logical_and(both_allels_mask, valid_controls_mask), axis=0) # Both allels, control
    valid_table[1,2] = num_valid_controls - valid_table[1,0] - valid_table[1,1] # 2 Major allels, case
    
    """
    # compute minor allele frequencies per SNP (MAFs)
    n_ma_per_snp = 2*(valid_table[0,0] + valid_table[1,0]) + valid_table[0,1] + valid_table[1,1]
    ma_freq_per_snp = np.divide(n_ma_per_snp, 2*n_valid_bearers_by_snp)
    assert(np.max(ma_freq_per_snp) <= 1 and np.min(ma_freq_per_snp >=0))
    ma_freq_per_snp = np.minimum(ma_freq_per_snp, (1.0 - ma_freq_per_snp)) 
    """

    # compute p-values
    case_ctrl_table = np.squeeze(np.sum(valid_table, axis=1)) # ~ nb of cases, nb of controls
    allel_mix_table = np.squeeze(np.sum(valid_table,axis=0)) # ~ nb of minor , mixed and major pairs

    valid_n_by_snp = np.sum(case_ctrl_table, axis=0)
    chisq_value=np.zeros(n_snps)
    expected = np.zeros(n_snps)

    for i in range(2):
        for j in range(3):
            expected = np.divide(np.multiply(case_ctrl_table[i], allel_mix_table[j]),valid_n_by_snp) # Independence assumption: e = nb_c_or_c * allel_mix_table / N
            nonzero_mask = np.where(expected!=0,True,False)
            observed = valid_table[i,j,nonzero_mask]
            zero_mask = np.logical_not(nonzero_mask)
            if (np.sum(zero_mask)!=len(expected)): # If expected has at least one nonzero value

                chisq_value[nonzero_mask] = chisq_value[nonzero_mask] + np.divide((observed - expected[nonzero_mask])**2, expected[nonzero_mask])
            
        
    pvalues = chi2.sf(chisq_value, 2)
    return pvalues

