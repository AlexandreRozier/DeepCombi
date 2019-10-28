import numpy as np 
from tqdm import tqdm
import h5py
import time

for chunksize in [100, 500, 1000]:
    chunked_elapsed_time = 0
    continous_elapsed_time = 0
    for i in tqdm(range (1,23)):
        with h5py.File('data/chromo_{}.mat'.format(i),'r') as f:
            print('Reading data...')
            start = time.time()
            data = f.get('X')[:]   
            print('Size of data: {} GB'.format(float(data.nbytes)/(1024*1024*1024)))
            continous_elapsed_time += float(time.time() - start)
        
        with h5py.File('data/chromo_{}_chunked.gz'.format(i),'w') as f2:
            print('generating chunked data')
            chunked_dset = f2.create_dataset("X", data=data, chunks=(chunksize, chunksize), compression="gzip")
        
        
        with h5py.File('data/chromo_{}_chunked.gz'.format(i),'r') as f3:
            print('reading chunked data')
            start = time.time()
            data = f3.get('X')[:]
            chunked_elapsed_time += float(time.time() - start)

    print ('Avg continous read time: {}'.format(continous_elapsed_time/22))
    print ('Avg chunked read time: {}'.format(chunked_elapsed_time/22))
