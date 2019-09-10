import os
import pickle
import h5py
import numpy as np 
from models import create_montaez_dense_model_2
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from helpers import char_matrix_to_featmat
from sklearn.model_selection import ParameterGrid
from parameters_complete import random_state, REAL_DATA_DIR, nb_of_nodes
from conftest import TEST_PERCENTAGE
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit

from Indices import Indices

import scipy.io
class TestLOTR(object):


    def test_train_networks(self):
        
        # Each node gets a set of chromosomes to process :D
        chroms_per_node = np.array_split(range(1,23), nb_of_nodes)

        for chrom in chroms_per_node[int(os.environ['SGE_TASK_ID'])-1]:

            # 1. Do hyperparam search on each chromosome and find parameters with BEST VAL ACCURACY
            
            params_space = {
                'epochs': [4],
                'patience': [10,30,50,100],
                'factor':np.linspace(0.1,0.8,9),
                'dropout_rate':np.linspace(0.1,0.5,5),
                'l1_reg': np.logspace(-6, -2, 5),
                'l2_reg': np.logspace(-6, -2, 5),
                'lr' : np.logspace(-4, -2, 3),        
            }

            grid = ParameterGrid(params_space)
            BUDGET = 2
            print("Testing {} % of the hp space".format(BUDGET/len(grid)*100))
            grid = random_state.choice(list(grid), BUDGET)


            f = h5py.File('data/chromo_{}.mat'.format(chrom),'r')
            data = f.get('X')
            fm = char_matrix_to_featmat(data, '3d')


            labels = scipy.io.loadmat('labels.mat')['y'][0]
            labels_0based = (labels+1)/2
            labels_cat = to_categorical(labels_0based)

            splitter =  StratifiedShuffleSplit(n_splits=1, test_size = TEST_PERCENTAGE, random_state=random_state)

            train_indices, test_indices = next(splitter.split(np.zeros(n_subjects), labels_0based))
            idx = Indices(train_indices, test_indices, None)

            best_acc = 0
            for g in grid:
                model = create_montaez_dense_model_2(g)

                history = model.fit(x=fm[idx.train],
                        y=labels_cat[idx.train],
                        validation_data=(fm[idx.test], labels_cat[idx.test]),
        
                        epochs=g['epochs'],
                        callbacks=[
                            TensorBoard(log_dir=os.path.join(REAL_DATA_DIR,'tb','chrom{}'.format(chrom)),
                                histogram_freq=3,
                                write_graph=False,
                                write_grads=True,
                                write_images=False),
                                ReduceLROnPlateau(monitor='val_loss', 
                                    factor=g['factor'], 
                                    patience=g['patience'], 
                                    mode='min'
                                ),
                        ],
                        verbose=1)
                val_acc = history.history['val_acc'][-1]

                # 2. Save this model  
                if val_acc > best_acc:
                    best_acc = val_acc
                    model.save(os.path.join(REAL_DATA_DIR,'models','chrom{}'.format(chrom)))
                    pickle.dump(g, os.path.join(REAL_DATA_DIR,'hyperparams','chrom{}'.format(chrom)))

                