
import os
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Indices import Indices
import h5py 
import torch

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l1
from keras.wrappers.scikit_learn import KerasClassifier

from parameters_complete import (Cs, IMG_DIR, PARAMETERS_DIR, DATA_DIR, SAVED_MODELS_DIR,
                                 TEST_DIR, n_total_snps, seed, p_pnorm_filter, filter_window_size, pnorm_feature_scaling, p_svm, noise_snps, inform_snps, top_k, ttbr, rep, thresholds,
                                 random_state, nb_of_nodes)
from models import DataGenerator, train_dummy_dense_model, create_montaez_dense_model, create_montaez_pytorch_model, create_dummy_pytorch_linear, create_dummy_conv_pytorch_model, ExConvNet
from helpers import EnforceNeg, generate_name_from_params, chi_square, postprocess_weights, compute_metrics, plot_pvalues, generate_syn_phenotypes, train_torch_model, evaluate_torch_model, train_keras_model
from combi import combi_method
import torch.multiprocessing as mp
from torch.utils.data.sampler import SubsetRandomSampler
from combi import classifier
import innvestigate
from keras.utils import to_categorical

true_pvalues = np.zeros((rep, n_total_snps), dtype=bool)
true_pvalues[:, int(noise_snps/2):int(noise_snps/2)+inform_snps] = True


class TestDNN(object):

    CONF_PATH = os.path.join(PARAMETERS_DIR, os.environ['SGE_TASK_ID'])



    
    def test_dense_model(self, h5py_data, labels, fm, labels_cat, indices, tmp_path):
        params = {
            'epochs': 20,
            'saved_model_path':os.path.join(tmp_path, 'dense.hdf5')
        }
        fm_ = fm("3d")
        
        features = fm_['0'][:]
        labels_c = labels_cat['0']
        labels_ = labels['0']
        best_model = train_dummy_dense_model(features, labels_c, indices, params)
        model_wo_sm = innvestigate.utils.keras.graph.model_wo_softmax(best_model)

        analyzer = innvestigate.analyzer.LRPZ(model_wo_sm)
        weights = analyzer.analyze(fm_['0'][:]).sum(0)

        top_indices_sorted = postprocess_weights(
            weights, top_k, filter_window_size, p_svm, p_pnorm_filter)
            
        complete_pvalues = chi_square(h5py_data['0'][:], labels_)

        fig, axes = plt.subplots(1, squeeze=True)

        plot_pvalues(complete_pvalues, top_indices_sorted, axes)
        axes.legend(["LRPZ, 300 subjects, dense; ttbr={}".format(ttbr)])
        fig.savefig(os.path.join(IMG_DIR,'dense-lrp.png'))
    


    def test_plot_dense(self,h5py_data, fm, indices, tmp_path):
        """ Compares efficiency of the combi method with several TTBR
        """
        ttbrs = [20, 6, 1, 0]
        h5py_data = h5py_data['0'][:]

        fig, axes = plt.subplots(len(ttbrs), sharex='col')
        fm_ = fm("3d")  

        
        for i, ttbr in enumerate(ttbrs):
            print('Using tbrr={}'.format(ttbr))
            labels = generate_syn_phenotypes(root_path=DATA_DIR, ttbr=ttbr)['0']
            l_cat = to_categorical((labels+1)/2)
            params = {
            'epochs': 20,
            'saved_model_path':os.path.join(tmp_path, 'dense.hdf5')

            }
            best_model = train_dummy_dense_model(fm_['0'][:],l_cat, indices, params)
            model_wo_sm = innvestigate.utils.keras.graph.model_wo_softmax(best_model)

            analyzer = innvestigate.analyzer.LRPZ(model_wo_sm)
            weights = analyzer.analyze(fm_['0'][:]).sum(0)

            print(weights)
            top_indices_sorted = postprocess_weights(
                weights, top_k, filter_window_size, p_svm, p_pnorm_filter)
                
            complete_pvalues = chi_square(h5py_data, labels)

            plot_pvalues(complete_pvalues, top_indices_sorted, axes[i])
            axes[i].legend(["LRPZ, 300 subjects, dense; ttbr={}".format(ttbr)])
            fig.savefig(os.path.join(IMG_DIR,'manhattan-dense-test.png'))
    

    
    def test_hp_params(self, fm, labels_0based, indices, hparams_array_path, output_path):
        
        ## PARAMETER GRID GENERATION
        params_array_per_node = []
        params_space = {
                'epochs':[3,5,10,20,40,80,100],
                'batch_size':[32],
                'lr':[10e-4, 10e-5],
                'l1_reg': np.logspace(10e-6, 10e-1, 6),
                'l2_reg': np.logspace(10e-6, 10e-1, 6),
                'dropout_rate':np.linspace(0.1,0.5, 3)
        }

        grid = ParameterGrid(params_space)
        n_by_node = math.ceil(sum(1.0 for _ in grid)/nb_of_nodes)

        i = 0
        j = 0
        tmp = []
        # Watch out: if not enough params, last nodes won't have any params
        for params in grid:
            # Node is full
            if j == n_by_node:
                j = 0
                i += 1
                params_array_per_node.append(tmp)
                tmp = []
            tmp.append(params)
            j+=1
        # Appends incomplete array:
        params_array_per_node.append(tmp) 
        assert len(params_array_per_node) == nb_of_nodes
        
        
        fm = fm('2d')
        
        def f(g):
            
            model = create_montaez_pytorch_model(g)
            val_losses = [train_keras_model(model,
                                    fm[str(i)][:],
                                    labels_0based[str(i)], 
                                    indices, 
                                    g)[1]['val_loss'] for i in tqdm(range(rep))]
            
            mean_loss_per_epoch = np.mean(val_losses, axis=1)
            best_epoch = np.argmin(mean_loss_per_epoch)
            g.pop('epochs', None)
            return best_epoch, mean_loss_per_epoch[best_epoch], g

        hparams_array = params_array_per_node[int(os.environ['SGE_TASK_ID'])]

        best_epoch, best_val_loss, params = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(f)(g) for g in hparams_array)).T
        results = pd.DataFrame(list(params))
        results['best_val_loss'] = best_val_loss
        results['best_epoch'] = best_epoch
        results.to_csv(output_path)


    def test_tpr_fwer_dense(self, h5py_data, labels, labels_cat, fm, indices):
        """ Compares combi vs dense curves
        """
        fig, axes = plt.subplots(2)
        fig.set_size_inches(18.5, 10.5)
        ax1, ax2 = axes

        ax1.set_ylabel('TPR')
        ax1.set_xlabel('FWER')
        ax1.set_ylim(0, 0.45)
        ax1.set_xlim(0, 0.1)

        ax2.set_ylabel('Precision')
        ax2.set_xlabel('TPR')

        def combi_compute_pvalues(d, fm, l):
           
            indices, pvalues = combi_method(d, fm, l, pnorm_feature_scaling,
                                            filter_window_size, top_k)
            pvalues_filled = np.ones(n_total_snps)
            pvalues_filled[indices] = pvalues
            del d, l
            return pvalues_filled

        def dense_compute_pvalues(i,d, fm, l_cat,l):
            params = {
                'epochs': 40,
                'l1_reg':1.584922,
                'l2_reg':2.511921,
                'lr':10e-3,
                'dropout_rate':0.3,
                'saved_model_path': os.path.join(SAVED_MODELS_DIR, 'dense-{}.h5py'.format(i))
            }
            model = create_montaez_dense_model(params)

            model, _ = train_keras_model(model, fm, labels_cat, indices, params)

            model_wo_sm = innvestigate.utils.keras.graph.model_wo_softmax(model)

            analyzer = innvestigate.analyzer.LRPZ(model_wo_sm)
            weights = analyzer.analyze(fm).sum(0)
            
            top_indices_sorted = postprocess_weights(
                weights, top_k, filter_window_size, p_svm, p_pnorm_filter)
            pvalues = chi_square(d[:, top_indices_sorted], l)
            pvalues_filled = np.ones(n_total_snps)
            pvalues_filled[top_indices_sorted] = pvalues
            del d,fm, l
            return pvalues_filled


        fm_2d = fm("2d")
        

        pvalues_per_run_combi = Parallel(n_jobs=-1, require='sharedmem')(delayed(
            combi_compute_pvalues)(h5py_data[str(i)][:],fm_2d[str(i)][:], labels[str(i)]) for i in tqdm(range(rep)))
        
        pvalues_per_run_dense = np.ones((rep, n_total_snps))
        for i in tqdm(range(rep)):
            pvalues_per_run_dense[i] = dense_compute_pvalues(i,h5py_data[str(i)][:], fm_2d[str(i)][:], labels_0based[str(i)],labels[str(i)])


        res_combi = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(
            pvalues_per_run_combi, true_pvalues, rep, threshold) for threshold in tqdm(thresholds)))
        
        res_dense = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(
            pvalues_per_run_dense, true_pvalues, rep, threshold) for threshold in tqdm(thresholds)))


        tpr_combi, enfr_combi, fwer_combi, precision_combi = res_combi.T
        tpr_dense, enfr_dense, fwer_dense, precision_dense = res_dense.T

        assert fwer_combi.max() <= 1 and fwer_combi.min() >= 0
        ax1.plot(fwer_combi, tpr_combi, '-o',
                 label='Combi method - ttbr={}'.format(ttbr))
        ax1.plot(fwer_dense, tpr_dense, '-x',
                 label='Dense method - ttbr={}'.format(ttbr))

        ax2.plot(tpr_combi, precision_combi, '-o',
                 label='Combi method - ttbr={}'.format(ttbr))
        ax2.plot(tpr_dense, precision_dense, '-x',
                 label='Dense method - ttbr={}'.format(ttbr))

        ax1.legend()
        ax2.legend()
        fig.savefig(os.path.join(IMG_DIR, 'tpr_fwer_dense_combi_comparison.png'), dpi=300)

    def test_svm_dnn_comparison(self, fm, labels, labels_cat, indices):
        """ Compares performance of SVM and DNN models
        """
        params = {
                'epochs': 40,
                'l1_reg':1.584922,
                'l2_reg':2.511921,
                'lr':10e-3,
                'dropout_rate':0.3,
                'batch_size': 32
        }
        fm_2d = fm("2d")
        fm_3d = fm("3d")

        svm_val_acc = np.zeros(rep-1)
        dnn_val_acc = np.zeros(rep-1)

        for i in tqdm(range(1, rep)):
            X_2d = fm_2d[str(i)][:]
            X_3d = fm_3d[str(i)][:]
            
            Y = labels[str(i)]
            Y_cat = labels_cat[str(i)]
            
            montaez_model = create_montaez_dense_model(params)
            # 1. Train dnn with optimal params
            montaez_model, history = train_keras_model(montaez_model, 
                                        X_3d, 
                                        Y_cat, 
                                        indices, 
                                        params)
            dnn_val_acc[i-1] = history['val_acc'][-1]
            

            # 2. Train svm with optimal params
            svm_model = classifier.fit(X_2d[indices.train], Y[indices.train])
            svm_val_acc[i-1] = svm_model.score(X_2d[indices.test], Y[indices.test])
            del X_2d, X_3d, Y, Y_cat
            
            
        
        # 3. Compare average val_acc on all syn datasets
        print('SVM mean val acc:{}; DNN mean val acc:{}'.format(np.mean(svm_val_acc), np.mean(dnn_val_acc)))

    # 4. If sufficient outperformance, try to plot the curve.

    def test_train_cnn(self,fm, labels_0based, indices):
        params = {
            'epochs':40,
            'lr':1e-4,
            'l2_reg':2.511921,
            'batch_size':32

        }
        model = ExConvNet()
        fm = fm('2d')['0'][:]
        data = torch.Tensor(fm)
        data = data.unsqueeze(1)
        l_0b = labels_0based['0']
        labels_0b = torch.LongTensor(l_0b)

        model, metrics = train_torch_model(model, data, labels_0b, indices, params)
        print(metrics['val_acc'])
