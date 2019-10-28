import numpy as np 
from tqdm import tqdm
import h5py
import time

chunked_elapsed_time = 0
continous_elapsed_time = 0
for i in tqdm(range (1,23)):
    with h5py.File('data/chromo_{}.mat'.format(i),'r') as f:
        print('Reading data...')
        start = time.time()
        data = f.get('X')[:]   
        print('Size of data: {} GB'.format(float(data.nbytes)/(1024*1024*1024)))
        continous_elapsed_time += float(time.time() - start)
    
    print('generating numpy data')
    np.save('data/chromo_{}.npy'.format(i), data, allow_pickle=True)    
    
    print('reading numpt data')
    start = time.time()
    data = np.load('data/chromo_{}.npy'.format(i), allow_pickle=True)[:]
    chunked_elapsed_time += float(time.time() - start)

print ('Avg continous read time: {}'.format(continous_elapsed_time/22))
print ('Avg np read time: {}'.format(chunked_elapsed_time/22))
