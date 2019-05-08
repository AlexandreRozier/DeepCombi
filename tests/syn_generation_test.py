import h5py
import hdf5storage
import pytest
import scipy
import os
import numpy as np
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

    maf = (np.sum(min_mask1, axis=0) + np.sum(min_mask2, axis=0))/(2*n_indiv)
    maf = np.minimum(maf, 1-maf)
    chrom_valid = chrom_valid[:, maf > 0.15, :]
    return chrom_valid


def generate_crohn_mat(c=6, n_replication=20, group_size=300, n_info_snps=20, n_noise_snps=10000):

    try:
        os.remove('/home/hx/Work/Masterarbeit/PythonImplementation/data/generated_data.txt')
    except FileNotFoundError:
        pass

    try:
        os.remove('/home/hx/Work/Masterarbeit/PythonImplementation/data/generated_labels.txt')
    except FileNotFoundError:
        pass    

    f = h5py.File('/home/hx/Work/Masterarbeit/PythonImplementation/data/chromo_01.mat','r')
    chrom1_full = f.get('X')
    chrom1_full = np.array(chrom1_full).T
    f.close()

    f2 = h5py.File('/home/hx/Work/Masterarbeit/PythonImplementation/data/chromo_02.mat','r')
    chrom2_full = f2.get('X')
    chrom2_full = np.array(chrom2_full).T
    f2.close()

    for i in tqdm(range(n_replication)):
        chrom1_full = np.roll(chrom1_full, group_size)
        chrom2_full = np.roll(chrom2_full, group_size)
        
        chrom1 = chrom1_full[:group_size]
        n_indiv, _ = chrom1.shape
        chrom1 = chrom1.reshape(n_indiv,-1,3)[:,:,:2]
        chrom1 = sanitize_chromosome(chrom1)

        
        chrom2 = chrom2_full[:group_size]
        n_indiv, _ = chrom2.shape
        chrom2 = chrom2.reshape(n_indiv,-1,3)[:,:,:2]
        chrom2 = sanitize_chromosome(chrom2)
        
        half_noise_size = int(n_noise_snps/2)
        data = np.concatenate((chrom2[:,:half_noise_size], chrom1[:,:n_info_snps], chrom2[:,half_noise_size:half_noise_size*2]), axis=1)
        # If the number of encoded SNPs is insufficient
        if data.shape[1] != n_info_snps + n_noise_snps:
            print("SKIPPING INDICES {}:{}".format(lower_bound,upper_bound))
            pass
        else:
            # Generate Informative  SNPs file
            with open('/home/hx/Work/Masterarbeit/PythonImplementation/data/generated_data.txt', 'a') as file:
                for i in tqdm(range(n_indiv)):
                    for j in range(n_info_snps+n_noise_snps):
                        all1 = data[i,j,0]
                        all2 = data[i,j,1]
                        if all1 < all2:
                            file.write("{}{} ".format(str(chr(all1)),str(chr(all2))))
                        else:
                            file.write("{}{} ".format(str(chr(all2)),str(chr(all1))))
                    file.write("\n")
                        
            
            # Generate Labels from UNIQUE SNP at position 9
        
            
            info_snp_idx = 9 
            info_snp = chrom1[:,info_snp_idx] # (n,2)
            lex_maj = info_snp.max() # (n,1)
            mask1 = np.where(info_snp[:,0] == lex_maj, True, False)
            mask2 = np.where(info_snp[:,1] == lex_maj, True, False)
            nb_major_allels = np.sum([mask1,mask2],axis=0) # (n,1)
            probabilities = np.power((1+np.exp(-c *(nb_major_allels - np.median(nb_major_allels)))),-1)
            random_vector = np.random.uniform(size=n_indiv)
            labels = np.where(probabilities>random_vector, "1","-1")

            with open('/home/hx/Work/Masterarbeit/PythonImplementation/data/generated_labels.txt', 'a') as file:
                for label in tqdm(labels):
                    file.write(label+"\n")
        
        
             

generate_crohn_mat()




# @Deprecated
def generate_genotyes_mat(self):
            
    f = h5py.File('/home/hx/Work/Masterarbeit/PythonImplementation/data/chromo_01.mat','r')
    chrom1 = f.get('X')
    chrom1 = np.array(chrom1).T[:100]
    f.close()

    f2 = h5py.File('/home/hx/Work/Masterarbeit/PythonImplementation/data/chromo_02.mat','r')
    chrom2 = f2.get('X')
    chrom2 = np.array(chrom2).T[:100]
    f2.close()


    allel1_1 = chrom1[:,::2]
    allel2_1 = chrom1[:,1::2]
    assert allel1_1.shape == allel2_1.shape
    print(allel1_1.shape)
    print(allel2_1.shape)


    allel1_2 = chrom2[:,::2]
    allel2_2 = chrom2[:,1::2]
    assert allel1_2.shape == allel2_2.shape 
    print(allel1_2.shape)
    print(allel2_2.shape)


    allel1 = np.concatenate((allel1_1,allel1_2), axis=1) 
    allel2 = np.concatenate((allel2_1,allel2_2), axis=1)
    hdf5storage.write({
        'allele1': allel1,
        'allele2': allel2
    },
    '.',
    '/home/hx/Work/Masterarbeit/PythonImplementation/data/genotypes.mat',
    matlab_compatible=True)

