import h5py 
import hdf5storage
import pytest 
import scipy 
import numpy as np

f = h5py.File('/home/hx/Work/Masterarbeit/PythonImplementation/data/chromo_01.mat','r')
chrom1 = f.get('X')
chrom1 = np.array(chrom1).T
f.close()

f2 = h5py.File('/home/hx/Work/Masterarbeit/PythonImplementation/data/chromo_02.mat','r')
chrom2 = f2.get('X')
chrom2 = np.array(chrom2).T
f2.close()


allel1_1 = chrom1[:,::2]
allel2_1 = chrom1[:,1::2]
assert allel1_1.shape == allel2_1.shape
print(allel1_1.shape)

allel1_2 = chrom2[:,::2]
allel2_2 = chrom2[:,1::2]
assert allel1_2.shape == allel2_2.shape 

allel1 = np.concatenate((allel1_1,allel1_2), axis=1) 
allel2 = np.concatenate((allel2_1,allel2_2), axis=1)
del allel1_1, allel1_2, allel2_1, allel2_2

hdf5storage.write({
     'allele1': allel1,
     'allele2': allel2
 },
 '.',
 '/home/hx/Work/Masterarbeit/PythonImplementation/data/genotypes.mat',
 matlab_compatible=True)
 