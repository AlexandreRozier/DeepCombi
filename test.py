import h5py
import os 


for disease in ['CD', 'BD','CAD','HT','RA','T1D','T2D']:
    for chrom in range(1,23):
        x = h5py.File(os.path.join('data',disease,'chromo_{}.h5py'.format(chrom)),'r').get('X')
        print('{}: {}: {}'.format(disease, chrom,x.shape))