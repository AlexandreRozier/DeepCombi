import h5py
import time

for i in range (1,23):
    with h5py.File('data/chromo_{}.mat'.format(i),'r') as f:
        start = time.time()
        data = f.get('X')[:]   
        print("Continous reading: {:.4f} s".format(time.time() - start))
    with h5py.File('data/chromo_{}_chunked.mat','w') as f2:
        chunked_dset = f2.create_dataset("X", data=data, chunks=True)
    
    with h5py.File('data/chromo_{}_chunked.mat','r') as f3:
        start = time.time()
        data = f3.get('X')[:]   
        print("Chunked reading: {:.4f} s".format(time.time() - start))