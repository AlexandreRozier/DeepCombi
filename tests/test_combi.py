import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time
import os
from helpers import chi_square, genomic_to_featmat, generate_syn_phenotypes, compute_metrics, plot_pvalues, char_matrix_to_featmat
from combi import combi_method, permuted_combi_method, toy_classifier
from tqdm import tqdm
from joblib import Parallel, delayed
from parameters_complete import thresholds, IMG_DIR, SYN_DATA_DIR, n_total_snps, inform_snps, noise_snps, \
    real_pnorm_feature_scaling
from parameters_complete import filter_window_size, top_k, ttbr , random_state, alpha_sig_toy



class TestCombi(object):
    

    def test_svm_accuracy(self, syn_genomic_data, syn_fm, syn_labels):

        bfm = syn_fm('2d')['0'][:]
        b_labels = syn_labels['0']
        toy_classifier.fit(bfm, b_labels)
        print("SVM score on Bettina's data: {}".format(toy_classifier.score(bfm, b_labels)))
    
        result = {}

        for ttbr in [6]:
            
            n = len(list(syn_genomic_data.keys()))
            train_accuracies = np.zeros(n)
            labels = generate_syn_phenotypes(root_path=SYN_DATA_DIR, tower_to_base_ratio=ttbr)
            for i, key in enumerate(list(syn_genomic_data.keys())):
                featmat = genomic_to_featmat()
                toy_classifier.fit(featmat, labels[key])
                train_accuracies[i] = toy_classifier.score(featmat, labels[key])
            result[str(ttbr)] = 'mean:{}; std: {}; best:{}'.format(np.mean(train_accuracies), np.std(train_accuracies), np.max(train_accuracies))
        
        print(result)


    def test_pvalues_generation(self, syn_genomic_data, syn_labels):
        pvalues = chi_square(syn_genomic_data['0'][:], syn_labels['0'])
        assert (-np.log10(pvalues)).max() > 9

    def test_pvalues_subset_generation(self, syn_genomic_data, syn_labels):
        syn_genomic_data = syn_genomic_data['0'][:]
        indices = random_state.randint(h5py_data.shape[1], size=top_k)
        pvalues = chi_square(h5py_data[:,indices,:], syn_labels['0'])
        assert(len(pvalues) == len( indices))
        assert(min(pvalues) >= 0 and max(pvalues) <=1)
    
    def test_combi(self, syn_genomic_data, syn_labels):
        syn_genomic_data = syn_genomic_data['0'][:]
        labels = syn_labels['0']
        fm = char_matrix_to_featmat(h5py_data, '2d', real_pnorm_feature_scaling)
        top_indices_sorted, pvalues, _ = combi_method(toy_classifier, h5py_data, fm, labels,
                                filter_window_size, top_k)
        print('PVALUES CHOSEN: {}'.format(top_indices_sorted))



    def test_compare_combi(self, syn_genomic_data, syn_labels):
        """ Compares efficiency of the combi method with several TTBR
        """
        ttbrs = [20, 6, 1, 0]
        b_data = syn_genomic_data['0'][:]
        b_labels = syn_labels['0']
        fig, axes = plt.subplots(len(ttbrs) + 1, sharex='col')

        complete_pvalues = chi_square(b_data, b_labels)

        top_indices_sorted, top_pvalues,_ = combi_method(b_data, b_labels,
                            filter_window_size, top_k)
        plot_pvalues(complete_pvalues, top_indices_sorted,top_pvalues, axes[0])
        axes[0].legend(["Matlab-generated data; Ttbr={}".format(6)])

        for i, ttbr in enumerate(ttbrs):
            print('Using tbrr={}'.format(ttbr))
            labels = generate_syn_phenotypes(root_path=SYN_DATA_DIR, tower_to_base_ratio=ttbr)['0']

            complete_pvalues = chi_square(b_data, labels)

            top_indices_sorted, top_pvalues,_ = combi_method(b_data, labels,
                                filter_window_size, top_k)
            plot_pvalues(complete_pvalues, top_indices_sorted, top_pvalues, axes[i+1])
            axes[i+1].legend(["Python-generated data; ttbr={}".format(ttbr)])
            fig.savefig(os.path.join(IMG_DIR,'combi_comparison.png'))
    
    
    def test_multiple_runs(self, syn_genomic_data):
        """ Compares efficiency of the combi method with several TTBR
        """
                    
        max_runs = 10
        fig, axes = plt.subplots(max_runs, sharex='col', sharey='col')
        fig.set_size_inches(18.5, 10.5)
        print('Using tbrr={}'.format(ttbr))
        labels = generate_syn_phenotypes(root_path=SYN_DATA_DIR, tower_to_base_ratio=ttbr)
        for i,key in enumerate(list(syn_genomic_data.keys())):
            if i >= max_runs:
                break
            data = syn_genomic_data[key][:]
            complete_pvalues = chi_square(data, labels[key])

            top_indices_sorted, top_pvalues,_ = combi_method(data, labels[key],
                                filter_window_size, top_k)
            plot_pvalues(complete_pvalues, top_indices_sorted, top_pvalues, axes[i])
        fig.savefig(os.path.join(IMG_DIR,'combi_multiple_runs.png'), dpi=100)
    
        


    def test_tpr_fwer_comparison(self, syn_genomic_data, rep):
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
        
        true_pvalues = np.zeros((rep, n_total_snps), dtype=bool)
        true_pvalues[:,int(noise_snps/2):int(noise_snps/2)+inform_snps] = True

        def p_compute_pvalues(data, labels):
            indices, pvalues,_ = combi_method(data, labels,
                                filter_window_size, top_k)
            pvalues_filled = np.ones(n_total_snps)
            pvalues_filled[indices] = pvalues
            del data, labels
            return pvalues_filled

        colors = cm.rainbow(np.linspace(0, 1, len(ttbr_list)))
        for j,ttbr in enumerate(tqdm(ttbr_list)):
            labels = generate_syn_phenotypes(root_path=SYN_DATA_DIR, tower_to_base_ratio=ttbr, n_info_snps=inform_snps,
                                             n_noise_snps=noise_snps)
            
            pvalues_per_run_combi = Parallel(n_jobs=-1, require='sharedmem')(delayed(p_compute_pvalues)(syn_genomic_data[str(i)][:], labels[str(i)]) for i in tqdm(range(rep)))
            pvalues_per_run_x2 = Parallel(n_jobs=-1, require='sharedmem')(delayed(chi_square)(syn_genomic_data[str(i)][:], labels[str(i)]) for i in tqdm(range(rep)))
           
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
    

    def test_tpr_fwer_k(self, syn_genomic_data, rep):
        fig, axes = plt.subplots(1)
        fig.set_size_inches(18.5, 10.5)
        ax1 = axes

        ax1.set_ylabel('TPR')
        ax1.set_xlabel('FWER')
        ax1.set_xlim(0,0.1)
     
        k_list = [10, 30, 50, 100, 500]
        true_pvalues = np.zeros((rep, n_total_snps), dtype=bool)
        true_pvalues[:,int(noise_snps/2):int(noise_snps/2)+inform_snps] = True

        labels = generate_syn_phenotypes(root_path=SYN_DATA_DIR, tower_to_base_ratio=ttbr, n_info_snps=inform_snps,
                                         n_noise_snps=noise_snps)

        
        # Try a couple of different ks 
        for k in tqdm(k_list):
            
            def f(data, labels):
    
                indices, pvalues, _ = combi_method(data, labels,
                                    filter_window_size, k)
                pvalues_filled = np.ones(n_total_snps)
                pvalues_filled[indices] = pvalues
                del data, labels
                return pvalues_filled
            
            pvalues_per_run_combi = Parallel(n_jobs=-1, require='sharedmem')(delayed(f)(syn_genomic_data[str(i)][:], labels[str(i)]) for i in range(rep))

            res_combi = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(pvalues_per_run_combi, true_pvalues, rep, threshold) for threshold in tqdm(thresholds)))

            tpr, enfr, fwer, precision = res_combi.T


            assert fwer.max() <=1 and fwer.min() >= 0
            ax1.plot(fwer,tpr,'-o', label='Combi method - k = {}'.format(k))

            
        ax1.legend()
        fig.savefig(os.path.join(IMG_DIR,'tpr_fwer_k.png'), dpi=300)
    

    def test_joblib_speedup(self, syn_labels, syn_fm, syn_genomic_data):
        syn_genomic_data = syn_genomic_data['0'][:]
        labels = syn_labels['0']
        fm = syn_fm('2d')['0'][:]
        n_permutations = 1000

        start_time = time.time()
        min_pvalues = np.zeros(n_permutations)
        for i in tqdm(range(n_permutations)):
            permuted_labels = random_state.permutation(labels)
            _, pvalues,_ = combi_method(h5py_data, fm, permuted_labels, filter_window_size, top_k)
            min_pvalues[i] = pvalues.min()
        t_star = np.quantile(min_pvalues, alpha_sig_toy)
        print('Sequential time: {} s'.format(time.time()-start_time))

        start_time = time.time()
        t_star = permuted_combi_method(h5py_data, fm, labels, n_permutations, alpha_sig_toy, filter_window_size, top_k)
        print('Parallel time: {} s'.format(time.time()-start_time))


    def test_tpr_fwer_ttbr(self, syn_genomic_data, rep, syn_true_pvalues):
        
        fig, axes = plt.subplots(1)
        fig.set_size_inches(18.5, 10.5)
        ax1 = axes

        ax1.set_ylabel('TPR')
        ax1.set_xlabel('FWER')
        ax1.set_xlim(0,0.1)

        ttbrs = [0.25, 2, 4, 6, 100]
        
        
        def f(data, labels):
    
            indices, pvalues, _ = combi_method(data, labels,
                                filter_window_size, top_k)
            pvalues_filled = np.ones(n_total_snps)
            pvalues_filled[indices] = pvalues
            del data, labels
            return pvalues_filled


        # Try a couple of different ttbr 
        for ttbr in tqdm(ttbrs):
            

            labels = generate_syn_phenotypes(root_path=SYN_DATA_DIR, tower_to_base_ratio=ttbr, n_info_snps=inform_snps,
                                             n_noise_snps=noise_snps)
            
            pvalues_per_run = Parallel(n_jobs=-1, require='sharedmem')(delayed(f)(syn_genomic_data[str(i)][:], labels[str(i)]) for i in range(rep))
            
            res_combi = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(pvalues_per_run, syn_true_pvalues, rep, threshold) for threshold in tqdm(thresholds)))

            tpr, enfr, fwer, precision = res_combi.T

            assert fwer.max() <=1 and fwer.min() >= 0
            ax1.plot(fwer,tpr,'-o', label='Combi method - ttbr = {}'.format(ttbr))
            
            
        ax1.legend()
        fig.savefig(os.path.join(IMG_DIR,'tpr_fwer_ttbr.png'), dpi=300)
    

    def test_permutations(self, syn_genomic_data, syn_fm, syn_labels, rep):
        true_pvalues = np.zeros((rep, n_total_snps), dtype=bool)
        true_pvalues[:,int(noise_snps/2):int(noise_snps/2)+inform_snps] = True

        syn_genomic_data = syn_genomic_data['0'][:]
        labels = syn_labels['0']
        fm = syn_fm('2d')['0'][:]
        n_permutations = 100
        t_star = permuted_combi_method(h5py_data, fm, labels, n_permutations, alpha_sig_toy, filter_window_size, top_k)
        pvalues = chi_square(h5py_data, labels)
        plt.scatter(range(len(pvalues)),-np.log10(pvalues), marker='x')
        plt.axhline(y=-np.log10(t_star), color='r', linestyle='-')
        assert t_star > 0
        assert t_star < pvalues.mean() # Tests t_star selectivity
        plt.savefig(os.path.join(IMG_DIR,'combi_permuted.png'), dpi=1000)
