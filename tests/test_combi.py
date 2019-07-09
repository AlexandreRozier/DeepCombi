import numpy as np
from helpers import chi_square, string_to_featmat, generate_syn_phenotypes, compute_metrics, plot_pvalues
from combi import combi_method, permuted_combi_method
from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import callbacks
from keras.models import load_model
import math
import os
import h5py
import copy
import innvestigate
import innvestigate.utils as iutils
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from joblib import Parallel, delayed
from sklearn import svm
from parameters_complete import thresholds, IMG_DIR, TEST_DIR, DATA_DIR, pnorm_feature_scaling, svm_rep, Cs, classy, n_total_snps, inform_snps, noise_snps
from parameters_complete import svm_epsilon, filter_window_size, p_pnorm_filter, top_k, ttbr as ttbr, rep, alpha_sig




true_pvalues = np.zeros((rep, n_total_snps), dtype=bool)
true_pvalues[:,int(noise_snps/2):int(noise_snps/2)+inform_snps] = True

class TestCombi(object):
    

    def test_svm_accuracy(self, h5py_data):
        with h5py.File(DATA_DIR+ '/bett_labels.h5py', 'r') as l, h5py.File(DATA_DIR+ '/bett_data.h5py', 'r') as d :
            b_labels =  l['X'][:]
            b_data =  d['X'][:]
        classifier = svm.LinearSVC(C=Cs, penalty='l2', tol=svm_epsilon, verbose=0, dual=True)
        bfm = string_to_featmat( b_data )

        classifier.fit(bfm, b_labels)
        print("SVM score on Bettina's data: {}".format(classifier.score(bfm, b_labels)))
    
        result = {}
        classifier = svm.LinearSVC(C=Cs, penalty='l2', tol=svm_epsilon, verbose=0, dual=True)
        
        for ttbr in [6]:
            
            n = len(list(h5py_data.keys()))
            train_accuracies = np.zeros(n)
            labels = generate_syn_phenotypes(root_path=DATA_DIR, ttbr=ttbr)
            for i, key in enumerate(list(h5py_data.keys())):
                featmat = string_to_featmat( h5py_data[key][:] )
                classifier.fit(featmat, labels[key])
                train_accuracies[i] = classifier.score(featmat, labels[key])
            result[str(ttbr)] = 'mean:{}; std: {}; best:{}'.format(np.mean(train_accuracies), np.std(train_accuracies), np.max(train_accuracies))
        
        print(result)


    def test_pvalues_generation(self, h5py_data, labels):
        pvalues = chi_square(h5py_data['0'][:], labels['0'])
        assert (-np.log10(pvalues)).max() > 9

    def test_pvalues_subset_generation(self, h5py_data, labels):
        h5py_data = h5py_data['0'][:]
        indices = np.random.randint(h5py_data.shape[1], size=top_k)
        pvalues = chi_square(h5py_data[:,indices,:], labels['0'])
        assert(len(pvalues) == len( indices))
        assert(min(pvalues) >= 0 and max(pvalues) <=1)
    
    def test_combi(self,h5py_data, labels):
        h5py_data = h5py_data['0'][:]
        labels = labels['0']
        top_indices_sorted, pvalues = combi_method(h5py_data, labels, pnorm_feature_scaling,
                                filter_window_size, top_k)
        print('PVALUES CHOSEN: {}'.format(top_indices_sorted))



    def test_compare_combi(self,h5py_data):
        """ Compares efficiency of the combi method with several TTBR
        """
        ttbrs = [20, 6, 1, 0]
        h5py_data = h5py_data['0'][:]

        fig, axes = plt.subplots(len(ttbrs) + 1, sharex='col')
        with h5py.File(DATA_DIR+ '/bett_labels.h5py', 'r') as l, h5py.File(DATA_DIR+ '/bett_data.h5py', 'r') as d :
            b_labels =  l['X'][:]
            b_data =  d['X'][:]
        
        complete_pvalues = chi_square(b_data, b_labels)

        top_indices_sorted, top_pvalues = combi_method(b_data, b_labels, pnorm_feature_scaling,
                            filter_window_size, top_k)
        self.plot_pvalues(complete_pvalues, top_indices_sorted,top_pvalues, axes[0])
        axes[0].legend(["Matlab-generated data; Ttbr={}".format(6)])

        for i, ttbr in enumerate(ttbrs):
            print('Using tbrr={}'.format(ttbr))
            labels = generate_syn_phenotypes(root_path=DATA_DIR, ttbr=ttbr)['0']

            complete_pvalues = chi_square(h5py_data, labels)

            top_indices_sorted, top_pvalues = combi_method(h5py_data, labels, pnorm_feature_scaling,
                                filter_window_size, top_k)
            self.plot_pvalues(complete_pvalues, top_indices_sorted, top_pvalues, axes[i+1])
            axes[i+1].legend(["Python-generated data; ttbr={}".format(ttbr)])
            fig.savefig(os.path.join(IMG_DIR,'combi_comparison.png'))
    
    
    def test_multiple_runs(self,h5py_data):
        """ Compares efficiency of the combi method with several TTBR
        """
                    
        max_runs = 10
        fig, axes = plt.subplots(max_runs, sharex='col', sharey='col')
        fig.set_size_inches(18.5, 10.5)
        print('Using tbrr={}'.format(ttbr))
        labels = generate_syn_phenotypes(root_path=DATA_DIR, ttbr=ttbr)
        for i,key in enumerate(list(h5py_data.keys())):
            if i >= max_runs:
                break
            data = h5py_data[key][:]
            complete_pvalues = chi_square(data, labels[key])

            top_indices_sorted, top_pvalues = combi_method(data, labels[key], pnorm_feature_scaling,
                                filter_window_size, top_k)
            self.plot_pvalues(complete_pvalues, top_indices_sorted, top_pvalues, axes[i])
        fig.savefig(os.path.join(IMG_DIR,'combi_multiple_runs.png'), dpi=100)
    
        


    def test_tpr_fwer_comparison(self, h5py_data):
        fig, axes = plt.subplots(2)
        fig.set_size_inches(18.5, 10.5)
        ax1, ax2 = axes

        ax1.set_ylabel('TPR')
        ax1.set_xlabel('FWER')
        ax1.set_ylim(0,0.45)
        ax1.set_xlim(0,0.1)

        ax2.set_ylabel('Precision')
        ax2.set_xlabel('TPR')
        ttbr_list = [0.25, 1, 1.5, 2, 4, 6, 100]
        
          
        def p_compute_pvalues(data, labels):
            indices, pvalues = combi_method(data, labels, pnorm_feature_scaling,
                                filter_window_size, top_k)
            pvalues_filled = np.ones(n_total_snps)
            pvalues_filled[indices] = pvalues
            del data, labels
            return pvalues_filled

        colors = cm.rainbow(np.linspace(0, 1, len(ttbr_list)))
        for j,ttbr in enumerate(tqdm(ttbr_list)):
            labels = generate_syn_phenotypes(ttbr=ttbr,root_path=DATA_DIR, n_info_snps=inform_snps, n_noise_snps=noise_snps)
            
            pvalues_per_run_combi = Parallel(n_jobs=-1, require='sharedmem')(delayed(p_compute_pvalues)(h5py_data[str(i)][:], labels[str(i)]) for i in tqdm(range(rep)))
            pvalues_per_run_x2 = Parallel(n_jobs=-1, require='sharedmem')(delayed(chi_square)(h5py_data[str(i)][:], labels[str(i)]) for i in tqdm(range(rep)))
           
            res_combi = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(pvalues_per_run_combi, true_pvalues, rep, threshold) for threshold in tqdm(thresholds)))
            res_x2 = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(pvalues_per_run_x2, true_pvalues, rep, threshold) for threshold in tqdm(thresholds)))

            tpr_combi, enfr_combi, fwer_combi, precision_combi = res_combi.T
            tpr_x2, enfr_x2, fwer_x2, precision_x2 = res_x2.T

            assert fwer_combi.max() <=1 and fwer_combi.min() >= 0
            ax1.plot(fwer_combi,tpr_combi,'-o', color=colors[j],label='Combi method - ttbr={}'.format(ttbr))
            ax1.plot(fwer_x2,tpr_x2,'-x',color=colors[j], label='RPVT method - ttbr={}'.format(ttbr))

            ax2.plot(tpr_combi,precision_combi,'-o',color=colors[j], label='Combi method - ttbr={}'.format(ttbr))
            ax2.plot(tpr_x2,precision_x2,'-x',color=colors[j], label='RPVT method - ttbr={}'.format(ttbr))
            
        ax1.legend()
        ax2.legend()
        fig.savefig(os.path.join(IMG_DIR,'tpr_fwer_comparison.png'), dpi=300)
    

    def test_tpr_fwer_k(self, h5py_data):
        fig, axes = plt.subplots(1)
        fig.set_size_inches(18.5, 10.5)
        ax1 = axes

        ax1.set_ylabel('TPR')
        ax1.set_xlabel('FWER')
        ax1.set_xlim(0,0.1)
     
        k_list = [10, 30, 50, 100, 500]
        
        labels = generate_syn_phenotypes(ttbr=ttbr,root_path=DATA_DIR, n_info_snps=inform_snps, n_noise_snps=noise_snps)

        
        # Try a couple of different ks 
        for k in tqdm(k_list):
            
            def f(data, labels):
    
                indices, pvalues = combi_method(data, labels, pnorm_feature_scaling,
                                    filter_window_size, k)
                pvalues_filled = np.ones(n_total_snps)
                pvalues_filled[indices] = pvalues
                del data, labels
                return pvalues_filled
            
            pvalues_per_run_combi = Parallel(n_jobs=-1, require='sharedmem')(delayed(f)(h5py_data[str(i)][:], labels[str(i)]) for i in range(rep))

            res_combi = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(pvalues_per_run_combi, true_pvalues, rep, threshold) for threshold in tqdm(thresholds)))

            tpr, enfr, fwer, precision = res_combi.T


            assert fwer.max() <=1 and fwer.min() >= 0
            ax1.plot(fwer,tpr,'-o', label='Combi method - k = {}'.format(k))

            
        ax1.legend()
        fig.savefig(os.path.join(IMG_DIR,'tpr_fwer_k.png'), dpi=300)
    

    def test_tpr_fwer_ttbr(self, h5py_data):
        
        fig, axes = plt.subplots(1)
        fig.set_size_inches(18.5, 10.5)
        ax1 = axes

        ax1.set_ylabel('TPR')
        ax1.set_xlabel('FWER')
        ax1.set_xlim(0,0.1)

        ttbrs = [0.25, 2, 4, 6, 100]
        
        
        def f(data, labels):
    
            indices, pvalues = combi_method(data, labels, pnorm_feature_scaling,
                                filter_window_size, top_k)
            pvalues_filled = np.ones(n_total_snps)
            pvalues_filled[indices] = pvalues
            del data, labels
            return pvalues_filled


        # Try a couple of different ttbr 
        for ttbr in tqdm(ttbrs):
            

            labels = generate_syn_phenotypes(ttbr=ttbr,root_path=DATA_DIR, n_info_snps=inform_snps, n_noise_snps=noise_snps)
            
            pvalues_per_run = Parallel(n_jobs=-1, require='sharedmem')(delayed(f)(h5py_data[str(i)][:], labels[str(i)]) for i in range(rep))
            
            res_combi = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(pvalues_per_run, true_pvalues, rep, threshold) for threshold in tqdm(thresholds)))

            tpr, enfr, fwer, precision = res_combi.T

            assert fwer.max() <=1 and fwer.min() >= 0
            ax1.plot(fwer,tpr,'-o', label='Combi method - ttbr = {}'.format(ttbr))
            
            
        ax1.legend()
        fig.savefig(os.path.join(IMG_DIR,'tpr_fwer_ttbr.png'), dpi=300)
    

    def test_permutations(self, h5py_data, labels):
  
        h5py_data = h5py_data['0'][:]
        labels = labels['0']
        t_star = permuted_combi_method(h5py_data, labels, rep, alpha_sig, pnorm_feature_scaling, filter_window_size, top_k)
        pvalues = chi_square(h5py_data, labels)
        plt.scatter(range(len(pvalues)),-np.log10(pvalues), marker='x')
        plt.axhline(y=-np.log10(t_star), color='r', linestyle='-')
        assert t_star > 0
        assert t_star < pvalues.mean() # Tests t_star selectivity
        plt.savefig(os.path.join(IMG_DIR,'combi_permuted.png'), dpi=1000)
