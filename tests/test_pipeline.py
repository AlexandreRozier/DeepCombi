import pandas as pd
import pickle 
import os
import scipy
import numpy as np
from tqdm import tqdm
from helpers import chi_square, char_matrix_to_featmat
from combi import deepcombi_method, combi_method
from parameters_complete import IMG_DIR, DATA_DIR,FINAL_RESULTS_DIR, real_p_pnorm_filter, p_svm, real_top_k, FINAL_RESULTS_DIR, filter_window_size, disease_IDs
from models import create_montaez_dense_model_2
from keras.models import load_model
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

class TestPipeline(object):


    def test_save_deepcombi_indices(self, real_h5py_data, real_labels):
        for disease in tqdm(disease_IDs):
            offset = 0
            selected_indices = []
            for chromo in tqdm(range(1,23)):
                model = load_model(os.path.join(FINAL_RESULTS_DIR, 'trained_models', disease, 'model{}.h5'.format(chromo)))

                data = real_h5py_data(disease,chromo)
                fm = char_matrix_to_featmat(data, '3d')
                labels = real_labels(disease)
                indices = offset + deepcombi_method(model, data, fm, labels, filter_window_size, real_p_pnorm_filter,
                                                    p_svm, top_k=real_top_k)[0]
                selected_indices.append(indices)
                offset += fm.shape[1]
                del data, fm
            
            np.save(os.path.join(FINAL_RESULTS_DIR, 'selected_indices', disease ), selected_indices)

    def test_save_combi_indices(self, real_h5py_data, real_labels):
        for disease in tqdm(disease_IDs):
            offset = 0
            selected_indices = []
            for chromo in tqdm(range(1, 23)):

                data = real_h5py_data(disease, chromo)
                fm_2D = char_matrix_to_featmat(data, '2d')
                fm_3D = char_matrix_to_featmat(data, '3d')
                labels = real_labels(disease)
                indices = offset + combi_method(data, fm_2D, labels, filter_window_size, real_p_pnorm_filter,
                                                    p_svm, top_k=real_top_k)[0]
                selected_indices.append(indices)
                offset += fm_3D.shape[1]
                del data, fm_2D, fm_3D

            np.save(os.path.join(FINAL_RESULTS_DIR, 'combi_selected_indices', disease), selected_indices)

    def test_read_data(self, real_pvalues):


        combined_labels = pd.Series()
        combined_combi_scores = pd.DataFrame()
        combined_deepcombi_scores = pd.DataFrame()
        combined_rpvt_scores = pd.DataFrame()

        for disease in tqdm(['CD', 'BD']):

            queries = pd.read_csv(os.path.join(DATA_DIR,'queries','{}.txt'.format(disease)), delim_whitespace=True)

            # Preselects indices of interest (at PEAKS, on pvalues smaller than 1e-5)
            offset = 0
            peaks_indices_genome = []
            raw_pvalues_genome = []

            for chromo in tqdm(range(1,23)):
                pvalues = real_pvalues(disease, chromo)
                raw_pvalues_genome +=  pvalues.tolist()
                pvalues_104 = np.ones(pvalues.shape)
                pvalues_104[pvalues < 1e-4] = pvalues[pvalues < 1e-4]
                peaks_indices, _ = scipy.signal.find_peaks(-np.log10(pvalues_104), distance=150)
                peaks_indices += offset
                offset += len(pvalues)
                peaks_indices_genome += peaks_indices.tolist()
                del pvalues, pvalues_104

            # Recreate queries
            raw_pvalues_genome = np.array(raw_pvalues_genome)
            peaks_indices_genome = np.array(peaks_indices_genome)

            # pvalues being < 10e-4 and belonging to a peak
            raw_pvalues_genome_peaks = raw_pvalues_genome[peaks_indices_genome]
            sorting_whole_genome_peaks_indices = np.argsort(raw_pvalues_genome_peaks)
            
            sorted_raw_pvalues_genome_peaks = pd.DataFrame(data={'pvalue':raw_pvalues_genome_peaks[sorting_whole_genome_peaks_indices]},
                                                           index=peaks_indices_genome[sorting_whole_genome_peaks_indices])

            # If the two queries match in size, add the rs_identifier field to our raw_pvalues. 
            # We can assign the identifiers thanks to the ordering of the 
            assert len(sorted_raw_pvalues_genome_peaks.index) == len(queries.index)
            sorted_raw_pvalues_genome_peaks['rs_identifier'] = queries.rs_identifier.tolist()

            # CREATE GROUND TRUTH LABELS
            # Note: tp_df DO NOT HAVE A VALID INDEX
            tp_df = pd.read_csv(os.path.join(DATA_DIR, 'results', '{}.txt'.format(disease)), delim_whitespace=True)
            tp_df = tp_df.rename(columns={"#SNP_A": "rs_identifier"})
            tp_df = sorted_raw_pvalues_genome_peaks.reset_index().merge(tp_df, on='rs_identifier',
                                                                        how='right').set_index('index')
            tp_df = tp_df.rename(columns={"pvalue_x": "pvalue"}).drop(columns=['pvalue_y'])

            pvalues_peak_labels = pd.Series(np.zeros(len(sorted_raw_pvalues_genome_peaks.index)),
                                       sorted_raw_pvalues_genome_peaks.index)
            pvalues_peak_labels.loc[np.intersect1d(sorted_raw_pvalues_genome_peaks.index, tp_df.index)] = 1
            combined_labels = pd.concat([combined_labels, pvalues_peak_labels])

            # COMBI
            selected_combi_indices = np.load(
                os.path.join(FINAL_RESULTS_DIR, 'selected_combi_indices', '{}.npy'.format(disease))).flatten()
            selected_combi_pvalues_peaks = sorted_raw_pvalues_genome_peaks.loc[
                np.intersect1d(selected_combi_indices, sorted_raw_pvalues_genome_peaks.index)]
            assert len(selected_combi_pvalues_peaks) > 0
            combined_combi_scores = pd.concat([combined_combi_scores, selected_combi_pvalues_peaks])

            # DeepCombi
            selected_deepcombi_indices = np.load(os.path.join(FINAL_RESULTS_DIR, 'selected_indices', '{}.npy'.format(disease))).flatten()
            selected_deepcombi_pvalues_peaks = sorted_raw_pvalues_genome_peaks.loc[np.intersect1d(selected_deepcombi_indices,sorted_raw_pvalues_genome_peaks.index)]
            assert len(selected_deepcombi_pvalues_peaks) > 0

            combined_deepcombi_scores = pd.concat([combined_deepcombi_scores, selected_deepcombi_pvalues_peaks ])

            # RPVT
            selected_rpvt = sorted_raw_pvalues_genome_peaks[sorted_raw_pvalues_genome_peaks.pvalue < 1e-5]
            combined_rpvt_scores = pd.concat([combined_rpvt_scores, selected_rpvt ])

        combi_fpr, combi_tpr, thresholds = roc_curve(
            combined_labels.loc[combined_combi_scores.index].values,
            -np.log10(combined_combi_scores.pvalue.values))
        combi_tp = combi_tpr * len(combined_combi_scores.index)
        combi_fp = combi_fpr * len(combined_combi_scores.index)

        deepcombi_fpr, deepcombi_tpr, thresholds =  roc_curve(combined_labels.loc[combined_deepcombi_scores.index].values, -np.log10(combined_deepcombi_scores.pvalue.values) )
        deepcombi_tp = deepcombi_tpr * len(combined_deepcombi_scores.index)
        deepcombi_fp = deepcombi_fpr * len(combined_deepcombi_scores.index)

        rpvt_fpr, rpvt_tpr, thresholds =  roc_curve(combined_labels.loc[combined_rpvt_scores.index].values, -np.log10(combined_rpvt_scores.pvalue.values) )
        rpvt_fp = rpvt_fpr * len(combined_rpvt_scores.index)
        rpvt_tp = rpvt_tpr * len(combined_rpvt_scores.index)

        fig = plt.figure()
        plt.plot(deepcombi_fp, deepcombi_tp, label='deepcombi')
        plt.plot(combi_fp, combi_tp, label='combi')
        plt.plot(rpvt_fp, rpvt_tp, label='rpvt')
        plt.xlabel('FP')
        plt.ylabel('TP')
        plt.legend()
        fig.savefig(os.path.join(IMG_DIR,'test.png'))

        #precision, recall, thresholds= precision_recall_curve(pvalues_labels, selected_combi_pvalues_peaks )
        #plt.scatter(precision, fpr)







