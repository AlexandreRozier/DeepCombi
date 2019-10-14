import pandas as pd
import pickle
import os
import scipy
import numpy as np
import sklearn
from tqdm import tqdm
from helpers import chi_square, char_matrix_to_featmat, postprocess_weights
from combi import deepcombi_method, combi_method, real_classifier
from parameters_complete import IMG_DIR, DATA_DIR, FINAL_RESULTS_DIR, real_pnorm_feature_scaling, real_p_pnorm_filter, \
    p_svm, real_top_k, FINAL_RESULTS_DIR, filter_window_size, disease_IDs
from models import create_montaez_dense_model_2
from keras.models import load_model
from sklearn.metrics import roc_curve, precision_recall_curve

import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt


class TestPipeline(object):


    def test_save_deepcombi_accuracies(self):
        for disease in tqdm(disease_IDs):
            scores = []
            for chrom in tqdm(range(1, 23)):
                data = pd.read_csv(os.path.join(FINAL_RESULTS_DIR,'csv_logs',disease,str(chrom)))
                scores += data.tail(1)['val_acc'].values.tolist()
            np.save(os.path.join(FINAL_RESULTS_DIR, 'accuracies', disease, 'deepcombi'), scores)

    def test_save_combi_scores_rm_and_indices(self, real_h5py_data, real_labels, real_idx):

        """ Extract indices gotten from combi
        """
        disease = disease_IDs[int(os.environ['SGE_TASK_ID']) - 1]
        offset = 0
        selected_indices = []
        total_raw_rm = np.empty((0,3))
        scores = []
        idx = real_idx(disease)
        labels = real_labels(disease)

        for chromo in tqdm(range(1,23)):
            data = real_h5py_data(disease, chromo)
            fm_2D = char_matrix_to_featmat(data, '2d', real_pnorm_feature_scaling)
            selected_idx, _, raw_rm = combi_method(real_classifier, data[idx.train], fm_2D[idx.train], labels[idx.train], filter_window_size, real_p_pnorm_filter,
                                                p_svm, top_k=real_top_k)
            real_classifier.fit(fm_2D[idx.train], labels[idx.train])
            scores.append(real_classifier.score(fm_2D[idx.test], labels[idx.test]))
               
            indices = offset + selected_idx

            selected_indices += indices.tolist()
            offset += int(fm_2D.shape[1]/3)
            total_raw_rm = np.append(total_raw_rm, raw_rm, axis=0)

            del data, fm_2D

        os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'combi_selected_indices'), exist_ok=True)
        os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'combi_raw_rm'), exist_ok=True)

        np.save(os.path.join(FINAL_RESULTS_DIR, 'combi_selected_indices', disease), selected_indices)
        np.save(os.path.join(FINAL_RESULTS_DIR, 'combi_raw_rm', disease ), total_raw_rm)
        np.save(os.path.join(FINAL_RESULTS_DIR, 'accuracies',disease,'combi'), scores)



    def test_svm(self):
        x = np.ones((10, 4))
        y = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        classifier = sklearn.svm.LinearSVC(C=2, penalty='l2', tol=3, verbose=0, dual=True)
        classifier.fit(x,y)
        classifier.coef_[0]

    def test_save_deepcombi_rm_and_indices(self, real_h5py_data, real_labels):
        """ Extract rm and selected indices obtained through deepcombi
        """
        disease = disease_IDs[int(os.environ['SGE_TASK_ID']) - 1]
        offset = 0
        selected_indices = []
        total_raw_rm = np.empty((0, 3))
        for chromo in tqdm(range(1, 23)):
            model = load_model(os.path.join(FINAL_RESULTS_DIR, 'trained_models', disease, 'model{}.h5'.format(chromo)))

            data = real_h5py_data(disease, chromo)
            fm = char_matrix_to_featmat(data, '3d', real_pnorm_feature_scaling)
            labels = real_labels(disease)
            idx, _, raw_rm = deepcombi_method(model, data, fm, labels, filter_window_size, real_p_pnorm_filter,
                                              p_svm, top_k=real_top_k)
            indices = offset + idx
            total_raw_rm = np.append(total_raw_rm, raw_rm, axis=0)
            selected_indices += indices.tolist()
            offset += fm.shape[1]
            del data, fm, model

        os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_selected_indices'), exist_ok=True)
        os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_raw_rm'), exist_ok=True)

        np.save(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_selected_indices', disease), selected_indices)
        np.save(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_raw_rm', disease), total_raw_rm)

    def test_generate_roc_recall_curves(self, real_pvalues):

        combined_labels = pd.Series()
        combined_combi_pvalues = pd.DataFrame()
        combined_deepcombi_pvalues = pd.DataFrame()
        combined_rpvt_scores = pd.DataFrame()
        combined_svm_scores = pd.Series()
        combined_deepcombi_scores = pd.Series()

        for disease in tqdm(disease_IDs):

            queries = pd.read_csv(os.path.join(DATA_DIR, 'queries', '{}.txt'.format(disease)), delim_whitespace=True)

            # Preselects indices of interest (at PEAKS, on pvalues smaller than 1e-4)
            offset = 0
            peaks_indices_genome = []
            raw_pvalues_genome = []

            for chromo in tqdm(range(1, 23)):
                pvalues = real_pvalues(disease, chromo)

                pvalues_104 = np.ones(pvalues.shape)
                pvalues_104[pvalues < 1e-4] = pvalues[pvalues < 1e-4]
                peaks_indices, _ = scipy.signal.find_peaks(-np.log10(pvalues_104), distance=150)
                peaks_indices += offset

                # BUGFIX for CAD
                if disease == 'CAD' and chromo != 9:
                    pvalues[pvalues < 1e-6] = 1

                offset += len(pvalues)
                raw_pvalues_genome += pvalues.tolist()
                peaks_indices_genome += peaks_indices.tolist()
                del pvalues, pvalues_104

            # Recreate queries
            raw_pvalues_genome = np.array(raw_pvalues_genome)
            peaks_indices_genome = np.array(peaks_indices_genome)

            # pvalues being < 10e-4 and belonging to a peak
            raw_pvalues_genome_peaks = raw_pvalues_genome[peaks_indices_genome]
            sorting_whole_genome_peaks_indices = np.argsort(raw_pvalues_genome_peaks)

            candidates_raw_pvalues_genome_peaks = pd.DataFrame(
                data={'pvalue': raw_pvalues_genome_peaks[sorting_whole_genome_peaks_indices]},
                index=peaks_indices_genome[sorting_whole_genome_peaks_indices])

            # If the two queries match in size, add the rs_identifier field to our raw_pvalues. 
            # We can assign the identifiers thanks to the ordering of the pvalues
            assert len(candidates_raw_pvalues_genome_peaks.index) == len(queries.index)
            candidates_raw_pvalues_genome_peaks['rs_identifier'] = queries.rs_identifier.tolist()

            # CREATE GROUND TRUTH LABELS
            tp_df = pd.read_csv(os.path.join(DATA_DIR, 'results', '{}.txt'.format(disease)), delim_whitespace=True)
            tp_df = tp_df.rename(columns={"#SNP_A": "rs_identifier"})
            tp_df = candidates_raw_pvalues_genome_peaks.reset_index().merge(tp_df,
                                                                            on='rs_identifier',
                                                                            how='right').set_index('index')
            tp_df = tp_df.rename(columns={"pvalue_x": "pvalue"}).drop(columns=['pvalue_y'])

            pvalues_peak_labels = pd.Series(data=np.zeros(len(candidates_raw_pvalues_genome_peaks.index)),
                                            index=candidates_raw_pvalues_genome_peaks.index)
            pvalues_peak_labels.loc[np.intersect1d(candidates_raw_pvalues_genome_peaks.index, tp_df.index)] = 1
            combined_labels = pd.concat([combined_labels, pvalues_peak_labels])

            # SVM WEIGHTS
            raw_svm_scores = np.load(os.path.join(FINAL_RESULTS_DIR, 'combi_raw_rm', '{}.npy'.format(disease)))
            processed_svm_scores = \
                postprocess_weights(raw_svm_scores, real_top_k, filter_window_size, p_svm, real_p_pnorm_filter)[1]
            processed_svm_scores = pd.Series(data=processed_svm_scores[candidates_raw_pvalues_genome_peaks.index],
                                             index=candidates_raw_pvalues_genome_peaks.index)
            combined_svm_scores = pd.concat([combined_svm_scores, processed_svm_scores])

            # DeepCombi WEIGHTS
            raw_deepcombi_scores = np.load(
                os.path.join(FINAL_RESULTS_DIR, 'deepcombi_raw_rm', '{}.npy'.format(disease)))
            processed_deepcombi_scores = \
                postprocess_weights(raw_deepcombi_scores, real_top_k, filter_window_size, p_svm, real_p_pnorm_filter)[1]
            processed_deepcombi_scores = pd.Series(
                data=processed_deepcombi_scores[candidates_raw_pvalues_genome_peaks.index],
                index=candidates_raw_pvalues_genome_peaks.index)
            combined_deepcombi_scores = pd.concat([combined_deepcombi_scores, processed_deepcombi_scores])

            # COMBI
            selected_combi_indices = np.load(
                os.path.join(FINAL_RESULTS_DIR, 'combi_selected_indices', '{}.npy'.format(disease))).flatten()
            selected_combi_pvalues = pd.Series(data=np.ones(len(candidates_raw_pvalues_genome_peaks.index)),
                                               index=candidates_raw_pvalues_genome_peaks.index)  # Build a all ones pvalues series
            idx = np.intersect1d(selected_combi_indices, candidates_raw_pvalues_genome_peaks.index)

            selected_combi_pvalues.loc[idx] = candidates_raw_pvalues_genome_peaks.loc[idx].pvalue
            assert len(selected_combi_pvalues.index) > 0
            combined_combi_pvalues = pd.concat([combined_combi_pvalues, selected_combi_pvalues])

            # DeepCombi
            selected_deepcombi_indices = np.load(
                os.path.join(FINAL_RESULTS_DIR, 'deepcombi_selected_indices', '{}.npy'.format(disease))).flatten()
            selected_deepcombi_pvalues = pd.Series(data=np.ones(len(candidates_raw_pvalues_genome_peaks.index)),
                                                   index=candidates_raw_pvalues_genome_peaks.index)  # Build a all ones pvalues series
            idx = np.intersect1d(selected_deepcombi_indices, candidates_raw_pvalues_genome_peaks.index)
            selected_deepcombi_pvalues.loc[idx] = candidates_raw_pvalues_genome_peaks.loc[idx].pvalue
            assert len(selected_deepcombi_pvalues.index) > 0
            combined_deepcombi_pvalues = pd.concat([combined_deepcombi_pvalues, selected_deepcombi_pvalues])

            # RPVT
            combined_rpvt_scores = pd.concat([combined_rpvt_scores, candidates_raw_pvalues_genome_peaks])

        # AUC stuff
        svm_fpr, svm_tpr, _ = roc_curve(combined_labels.values,
                                        combined_svm_scores.values)
        svm_tp = svm_tpr * (combined_labels.values == 1).sum()
        svm_fp = svm_fpr * (combined_labels.values != 1).sum()

        dc_rm_fpr, dc_rm_tpr, _ = roc_curve(combined_labels.values,
                                            combined_deepcombi_scores.values)
        dc_rm_tp = dc_rm_tpr * (combined_labels.values == 1).sum()
        dc_rm_fp = dc_rm_fpr * (combined_labels.values != 1).sum()

        combi_fpr, combi_tpr, _ = roc_curve(combined_labels.values,
                                            -np.log10(combined_combi_pvalues.values))
        combi_tp = combi_tpr * (combined_labels.values == 1).sum()
        combi_fp = combi_fpr * (combined_labels.values != 1).sum()

        deepcombi_fpr, deepcombi_tpr, _ = roc_curve(
            combined_labels.values,
            -np.log10(combined_deepcombi_pvalues.values))
        deepcombi_tp = deepcombi_tpr * (combined_labels.values == 1).sum()
        deepcombi_fp = deepcombi_fpr * (combined_labels.values != 1).sum()

        rpvt_fpr, rpvt_tpr, _ = roc_curve(combined_labels.values,
                                          -np.log10(combined_rpvt_scores.pvalue.values))
        rpvt_tp = rpvt_tpr * (combined_labels.values == 1).sum()
        rpvt_fp = rpvt_fpr * (combined_labels.values != 1).sum()

        fig = plt.figure()
        plt.plot(deepcombi_fp, deepcombi_tp, label='deepcombi')
        plt.plot(combi_fp, combi_tp, label='combi')
        plt.plot(rpvt_fp, rpvt_tp, label='rpvt')
        plt.plot(svm_fp, svm_tp, label='SVM weights')
        plt.plot(dc_rm_fp, dc_rm_tp, label='DeepCOMBI weights')
        plt.xlabel('FP')
        plt.ylabel('TP')
        plt.xlim(0, 80)
        plt.ylim(0, 40)
        plt.legend()
        fig.savefig(os.path.join(FINAL_RESULTS_DIR, 'plots', 'auc.png'))

        # Precision recall stuff

        combi_precision, combi_recall, thresholds = precision_recall_curve(combined_labels.values,
                                                                           -np.log10(combined_combi_pvalues.values))
        combi_tp = combi_recall * (combined_labels.values == 1).sum()

        deepcombi_precision, deepcombi_recall, _ = precision_recall_curve(
            combined_labels.values,
            -np.log10(combined_deepcombi_pvalues.values))
        deepcombi_tp = deepcombi_recall * (combined_labels.values == 1).sum()

        rpvt_precision, rpvt_recall, _ = precision_recall_curve(combined_labels.values,
                                                                -np.log10(combined_rpvt_scores.pvalue.values))
        rpvt_tp = rpvt_recall * (combined_labels.values == 1).sum()

        svm_precision, svm_recall, _ = precision_recall_curve(combined_labels.values,
                                                              combined_svm_scores.values)
        svm_tp = svm_recall * (combined_labels.values == 1).sum()

        dc_rm_precision, dc_rm_recall, _ = precision_recall_curve(combined_labels.values,
                                                                  combined_deepcombi_scores.values)
        dc_rm_tp = dc_rm_recall * (combined_labels.values == 1).sum()

        fig = plt.figure()
        plt.plot(deepcombi_tp, deepcombi_precision, label='deepcombi')
        plt.plot(combi_tp, combi_precision, label='combi')
        plt.plot(rpvt_tp, rpvt_precision, label='rpvt')
        plt.plot(svm_tp, svm_precision, label='SVM')
        plt.plot(dc_rm_tp, dc_rm_precision, label='DeepCOMBI')
        plt.xlabel('TP')
        plt.ylabel('Precision')
        plt.xlim(0, 40)

        plt.legend()
        fig.savefig(os.path.join(FINAL_RESULTS_DIR, 'plots', 'precision-tp.png'))


    def test_print_accuracies(self):
        for disease in disease_IDs:
            combi_acc = np.load(os.path.join(FINAL_RESULTS_DIR, 'accuracies',disease,'combi.npy'))            
            dc_acc = np.load(os.path.join(FINAL_RESULTS_DIR, 'accuracies',disease,'deepcombi.npy'))
            print(disease)
            print('COMBI val_acc: {}'.format(combi_acc))            
            print('DeepCOMBI val_acc: {}'.format(dc_acc))