import pandas as pd
import pickle 
import os
import scipy
import numpy as np
from tqdm import tqdm
from helpers import chi_square, char_matrix_to_featmat
from combi import deepcombi_method
from parameters_complete import DATA_DIR,FINAL_RESULTS_DIR, real_p_pnorm_filter, p_svm, real_top_k, FINAL_RESULTS_DIR, filter_window_size, disease_IDs
from models import create_montaez_dense_model_2
from keras.models import load_model
from sklearn.metrics import roc_curve, precision_recall_curve

class TestPipeline(object):


    def test_save_deepcombi_indices(self, real_h5py_data, real_labels, real_labels_cat):
        for disease in tqdm(disease_IDs):
            offset = 0
            selected_indices = []
            for chromo in tqdm(range(1,23)):
                model = load_model(os.path.join(FINAL_RESULTS_DIR, 'trained_models', disease, 'model{}.h5'.format(chromo)))

                data = real_h5py_data(disease,chromo)
                fm = char_matrix_to_featmat(data, '3d')
                labels = real_labels(disease)
                labels_cat = real_labels_cat(disease)
                indices = offset + deepcombi_method(model, data, fm, labels, labels_cat, filter_window_size, real_p_pnorm_filter, p_svm, top_k=real_top_k)[0]
                selected_indices.append(indices)
                offset += fm.shape[1]
                del data, fm
            
            np.save(os.path.join(FINAL_RESULTS_DIR, 'selected_indices', disease ), selected_indices)


    def test_read_data(self, real_pvalues):

        
        for disease in tqdm(disease_IDs):

            queries = pd.read_csv(os.path.join(DATA_DIR,'queries','{}.txt'.format(disease)))

            # Preselects indices of interest (at PEAKS, on pvalues smaller than 1e-5)
            offset = 0
            peaks_indices_genome = np.array()
            raw_pvalues_genome = np.array()

            for chromo in tqdm(range(1,23)):
                pvalues = real_pvalues(disease, chromo)
                raw_pvalues_genome = np.append(raw_pvalues_genome, pvalues)
                pvalues_104 = pvalues[pvalues < 1e-4]
                peaks_indices = scipy.signal.find_peaks(pvalues_104, width=150)
                peaks_indices += offset
                peaks_indices_genome = np.append(peaks_indices_genome, peaks_indices)
                del pvalues, pvalues_104

            # Recreate queries
            raw_pvalues_genome_peaks = raw_pvalues_genome[peaks_indices_genome]
            sorting_whole_genome_peaks_indices = np.argsort(raw_pvalues_genome_peaks)
            
            sorted_raw_pvalues_genome_peaks = pd.Series(raw_pvalues_genome_peaks[sorting_whole_genome_peaks_indices], sorting_whole_genome_peaks_indices)

            # If the two queries match in size, add the rs_identifier field to our raw_pvalues. 
            # We can assign the identifiers thanks to the ordering of the 
            assert len(sorted_raw_pvalues_genome_peaks.index) == len(queries.index)
            sorted_raw_pvalues_genome_peaks.rs_identifier = queries.rs_identifier 

            # Combi
            selected_indices = np.load(os.path.join(FINAL_RESULTS_DIR, 'selected_indices', disease))
           

            # Index-based selection (iloc would be incorrect here!).
            selected_combi_pvalues_peaks   = sorted_raw_pvalues_genome_peaks[selected_indices] 

            
            # Note: tp_df DO NOT HAVE A VALID INDEX
            tp_df = pd.read_csv(os.path.join(DATA_DIR,'results','{}.txt'.format(disease)))
            tp_df = tp_df.rename(columns={"SNP_A": "rs_identifier"})
            tp_df.merge(queries, on='key', left_index=True, how='left') # based on the fact that queries contains all values of results

            #combi_tp_df = selected_combi_pvalues_peaks.join(tp_df, on=['rs_identifier'], how='left')
            #combi_fp_df = selected_combi_pvalues_peaks[pd.Index.difference(selected_combi_pvalues_peaks.index, combi_tp_df.index)]
            #combi_fn_df = tp_df[pd.Index.difference(tp_df.index, selected_combi_pvalues_peaks.index)]
            
            pvalues_labels = np.zeros(len(selected_combi_pvalues_peaks))
            pvalues_labels[tp_df.index] = 1

            fpr, tpr, thresholds =  (pvalues_labels, selected_combi_pvalues_peaks )

            precision, recall, thresholds= precision_recall_curve(pvalues_labels, selected_combi_pvalues_peaks )





