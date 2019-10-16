import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy

from sklearn.metrics import roc_curve, precision_recall_curve

import seaborn
import pandas as pd
import innvestigate
from combi import real_classifier
from helpers import char_matrix_to_featmat, postprocess_weights
from models import best_params_montaez_2, create_montaez_dense_model_2
from innvestigate import utils as iutils

from tqdm import tqdm
import numpy as np

from parameters_complete import disease_IDs, FINAL_RESULTS_DIR, real_pnorm_feature_scaling, real_top_k, \
    filter_window_size, p_svm, real_p_pnorm_filter, IMG_DIR, DATA_DIR, ROOT_DIR


class TestWTCCCPlots(object):
    def test_lrp(self, real_h5py_data, real_labels, real_labels_0based, real_pvalues, real_labels_cat, real_idx, rep,
                 tmp_path):

        fig, axes = plt.subplots(5, 1, sharex='col')

        h5py_data = real_h5py_data(1)

        x_2d = char_matrix_to_featmat(h5py_data, '2d', real_pnorm_feature_scaling)
        x_3d = char_matrix_to_featmat(h5py_data, '3d', real_pnorm_feature_scaling)

        g = best_params_montaez_2
        g['n_snps'] = x_3d.shape[1]

        model = create_montaez_dense_model_2(g)
        model.fit(x=x_3d[real_idx.train],
                  y=real_labels_cat[real_idx.train],
                  validation_data=(x_3d[real_idx.test], real_labels_cat[real_idx.test]),
                  epochs=g['epochs']
                  )

        model = iutils.keras.graph.model_wo_softmax(model)
        analyzer = innvestigate.analyzer.LRPAlpha1Beta0(model)
        weights = np.absolute(analyzer.analyze(x_3d).sum(0))

        _, postprocessed_weights = postprocess_weights(
            weights, real_top_k, filter_window_size, p_svm, real_p_pnorm_filter)

        complete_pvalues = real_pvalues('CD', 1)
        axes[0].plot(-np.log10(complete_pvalues))
        axes[0].set_title('RPV')

        # Plot distribution of relevance
        axes[1].plot(np.absolute(weights).reshape(-1, 3).sum(1))
        axes[1].set_title('Absolute relevance')

        # Plot distribution of postprocessed vectors
        axes[2].plot(postprocessed_weights)
        axes[2].set_title('Postprocessed relevance')

        # Plot distribution of svm weights
        svm_weights = real_classifier.fit(x_2d, real_labels).coef_
        axes[3].plot(np.absolute(svm_weights).reshape(-1, 3).sum(1))
        axes[3].set_title('Absolute SVM weight')

        _, postprocessed_svm_weights = postprocess_weights(
            np.absolute(svm_weights), real_top_k, filter_window_size, p_svm, real_p_pnorm_filter)

        axes[4].plot(postprocessed_svm_weights)
        axes[4].set_title('Postprocessed SVM weight')

        fig.savefig(os.path.join(IMG_DIR, 'manhattan-real.png'))






    def test_generate_per_disease_manhattan_plots(self, real_pvalues, chrom_length):
        """ Plot manhattan figures of rpvt VS deepcombi, for one specific disease
        """
        for disease_id in tqdm(['CD']):

            chromos = range(1, 23)
            offsets = np.zeros(len(chromos) + 1, dtype=np.int)
            middle_offset_history = np.zeros(len(chromos), dtype=np.int)

            chrom_fig, axes = plt.subplots(5, 1, sharex='col')
            chrom_fig.tight_layout()
            chrom_fig.set_size_inches(18.5, 10.5)

            raw_pvalues_ax = axes[0]
            c_selected_pvalues_ax = axes[1]
            c_rm_ax = axes[2]
            deepc_selected_pvalues_ax = axes[3]
            deepc_rm_ax = axes[4]

            # Set title
            raw_pvalues_ax.set_title('Raw P-Values', fontsize=16)
            deepc_selected_pvalues_ax.set_title('DeepCOMBI Method', fontsize=16)
            c_selected_pvalues_ax.set_title('COMBI Method', fontsize=16)
            c_rm_ax.set_title("COMBI's Relevance Mapping", fontsize=16)
            deepc_rm_ax.set_title("DeepCOMBI's Relevance Mapping", fontsize=16)
            deepc_rm_ax.set_xlabel('Chromosome number')

            # Set labels
            raw_pvalues_ax.set_ylabel('-log_10(pvalue)')
            deepc_selected_pvalues_ax.set_ylabel('-log_10(pvalue)')
            c_selected_pvalues_ax.set_ylabel('-log_10(pvalue)')
            c_rm_ax.set_ylabel('SVM weights in %')
            deepc_rm_ax.set_ylabel('LRP relevance mapping in %')
            # Actually plot stuff
            top_indices_deepcombi = np.load(
                os.path.join(FINAL_RESULTS_DIR, 'deepcombi_selected_indices', '{}.npy'.format(disease_id)))
            top_indices_combi = np.load(
                os.path.join(FINAL_RESULTS_DIR, 'combi_selected_indices', '{}.npy'.format(disease_id)))


            complete_pvalues = []
            n_snps = np.zeros(22)

            svm_scaled_weights = np.load(
                os.path.join(FINAL_RESULTS_DIR, 'combi_scaled_rm', '{}.npy'.format(disease_id)))
            dc_scaled_weights = np.load(
                os.path.join(FINAL_RESULTS_DIR, 'deepcombi_scaled_rm', '{}.npy'.format(disease_id)))

            for i, chromo in enumerate(chromos):

                raw_pvalues = real_pvalues(disease_id, chromo)

                if disease_id == 'CAD' and chromo != 9:
                    raw_pvalues[raw_pvalues < 1e-6] = 1

                complete_pvalues += raw_pvalues.tolist()

                n_snps[i] = len(raw_pvalues)
                offsets[i + 1] = offsets[i] + n_snps[i]
                middle_offset_history[i] = offsets[i] + int(n_snps[i] / 2)
                svm_scaled_weights[offsets[i]:offsets[i+1]]*=np.sqrt(chrom_length(disease_id, chromo))
                dc_scaled_weights[offsets[i]:offsets[i+1]]*=np.sqrt(chrom_length(disease_id, chromo))

            complete_pvalues = np.array(complete_pvalues).flatten()

            combi_selected_pvalues = np.ones(len(complete_pvalues))
            combi_selected_pvalues[top_indices_combi] = complete_pvalues[top_indices_combi]

            deepcombi_selected_pvalues = np.ones(len(complete_pvalues))
            deepcombi_selected_pvalues[top_indices_deepcombi] = complete_pvalues[top_indices_deepcombi]

            informative_idx = np.argwhere(complete_pvalues < 1e-5)


            # Color
            color = np.zeros((len(complete_pvalues), 3))
            alt = True
            for i, offset in enumerate(offsets[:-1]):
                color[offset:offsets[i + 1]] = [0, 0, 0.7] if alt else [0.4, 0.4, 0.8]
                alt = not alt
            color[informative_idx] = [0, 1, 0]

            # Plot
            raw_pvalues_ax.scatter(range(len(complete_pvalues)), -np.log10(complete_pvalues), c=color, marker='x')
            deepc_selected_pvalues_ax.scatter(range(len(complete_pvalues)), -np.log10(deepcombi_selected_pvalues),
                                              c=color, marker='x')
            c_selected_pvalues_ax.scatter(range(len(complete_pvalues)), -np.log10(combi_selected_pvalues),
                                          c=color, marker='x')
            c_rm_ax.scatter(range(len(complete_pvalues)), svm_scaled_weights * 100, c=color, marker='x')
            deepc_rm_ax.scatter(range(len(complete_pvalues)), dc_scaled_weights * 100, c=color, marker='x')

            # Set ticks
            plt.setp(raw_pvalues_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
            plt.setp(deepc_selected_pvalues_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
            plt.setp(c_selected_pvalues_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
            plt.setp(c_rm_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
            plt.setp(deepc_rm_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
            chrom_fig.canvas.set_window_title(disease_id)

            chrom_fig.savefig(os.path.join(FINAL_RESULTS_DIR, 'plots', '{}-manhattan.png'.format(disease_id)))

    def test_generate_lrp_plots(self, real_pvalues, chrom_length):
        """ Plot manhattan figures of rpvt VS deepcombi, for one specific disease
        """
        disease_id = 'CD'

        chromos = range(1, 23)
        offsets = np.zeros(len(chromos) + 1, dtype=np.int)
        middle_offset_history = np.zeros(len(chromos), dtype=np.int)

        chrom_fig, axes = plt.subplots(3, 1, sharex='col')
        chrom_fig.tight_layout()

        raw_pvalues_ax = axes[0]
        c_rm_ax = axes[1]
        deepc_rm_ax = axes[2]

        # Set title
        raw_pvalues_ax.set_title('Raw P-Values', fontsize=16)
        c_rm_ax.set_title("COMBI's Relevance Mapping", fontsize=16)
        deepc_rm_ax.set_title("DeepCOMBI's Relevance Mapping", fontsize=16)
        deepc_rm_ax.set_xlabel('Chromosome number')

        # Set labels
        raw_pvalues_ax.set_ylabel('-log_10(pvalue)')
        c_rm_ax.set_ylabel('SVM weights in %')
        deepc_rm_ax.set_ylabel('LRP relevance mapping in %')
        # Actually plot stuff
        top_indices_deepcombi = np.load(
            os.path.join(FINAL_RESULTS_DIR, 'deepcombi_selected_indices', '{}.npy'.format(disease_id)))
        top_indices_combi = np.load(
            os.path.join(FINAL_RESULTS_DIR, 'combi_selected_indices', '{}.npy'.format(disease_id)))


        complete_pvalues = []
        n_snps = np.zeros(22)
        for i, chromo in enumerate(chromos):

            raw_pvalues = real_pvalues(disease_id, chromo)

            if disease_id == 'CAD' and chromo != 9:
                raw_pvalues[raw_pvalues < 1e-6] = 1

            complete_pvalues += raw_pvalues.tolist()

            n_snps[i] = len(raw_pvalues)
            offsets[i + 1] = offsets[i] + n_snps[i]
            middle_offset_history[i] = offsets[i] + int(n_snps[i] / 2)

        complete_pvalues = np.array(complete_pvalues).flatten()

        svm_scaled_weights = np.load(os.path.join(FINAL_RESULTS_DIR, 'combi_scaled_rm', '{}.npy'.format(disease_id)))
        dc_scaled_weights = np.load(
            os.path.join(FINAL_RESULTS_DIR, 'deepcombi_scaled_rm', '{}.npy'.format(disease_id)))

        # Account for the chromosom's length
        for i in range(22):
            svm_scaled_weights[offsets[i]:offsets[i+1]] = svm_scaled_weights[offsets[i]:offsets[i+1]] * np.sqrt(chrom_length(disease_id, i + 1))
            dc_scaled_weights[offsets[i]:offsets[i+1]] = dc_scaled_weights[offsets[i]:offsets[i+1]]*np.sqrt(chrom_length(disease_id, i+1))

        combi_selected_pvalues = np.ones(len(complete_pvalues))
        combi_selected_pvalues[top_indices_combi] = complete_pvalues[top_indices_combi]

        deepcombi_selected_pvalues = np.ones(len(complete_pvalues))
        deepcombi_selected_pvalues[top_indices_deepcombi] = complete_pvalues[top_indices_deepcombi]

        informative_idx = np.argwhere(complete_pvalues < 1e-5)

        # Color
        color = np.zeros((len(complete_pvalues), 3))
        alt = True
        for i, offset in enumerate(offsets[:-1]):
            color[offset:offsets[i + 1]] = [0, 0, 0.7] if alt else [0.4, 0.4, 0.8]
            alt = not alt
        color[informative_idx] = [0, 1, 0]

        # Plot
        raw_pvalues_ax.scatter(range(len(complete_pvalues)), -np.log10(complete_pvalues), c=color, marker='x')
        c_rm_ax.scatter(range(len(complete_pvalues)), svm_scaled_weights , c=color, marker='x')
        deepc_rm_ax.scatter(range(len(complete_pvalues)), dc_scaled_weights , c=color, marker='x')

        # Set ticks
        plt.setp(raw_pvalues_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
        plt.setp(c_rm_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
        plt.setp(deepc_rm_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
        chrom_fig.canvas.set_window_title(disease_id)

        chrom_fig.savefig(os.path.join(FINAL_RESULTS_DIR, 'plots', '{}-lrp.png'.format(disease_id)))

    def test_generate_global_manhattan_plots(self, real_pvalues):
        """ Plot manhattan figures of rpvt VS deepcombi, for one specific disease
        """
        chrom_fig, axes = plt.subplots(7, 3, sharex='col')
        chrom_fig.tight_layout()
        chrom_fig.set_size_inches(18.5, 10.5)
        axes[-1][0].set_xlabel('Chromosome number')
        axes[-1][1].set_xlabel('Chromosome number')
        axes[-1][2].set_xlabel('Chromosome number')

        for l, disease_id in tqdm(enumerate(disease_IDs)):

            raw_pvalues_ax = axes[l][0]
            c_selected_pvalues_ax = axes[l][1]
            deepc_selected_pvalues_ax = axes[l][2]

            chromos = range(1, 23)
            offsets = np.zeros(len(chromos) + 1, dtype=np.int)
            middle_offset_history = np.zeros(len(chromos), dtype=np.int)

            # Set title
            raw_pvalues_ax.set_title('Raw P-Values', fontsize=16)
            deepc_selected_pvalues_ax.set_title('DeepCOMBI Method', fontsize=16)
            c_selected_pvalues_ax.set_title('COMBI Method', fontsize=16)

            # Set labels
            raw_pvalues_ax.set_ylabel('-log_10(pvalue)')
            deepc_selected_pvalues_ax.set_ylabel('-log_10(pvalue)')
            c_selected_pvalues_ax.set_ylabel('-log_10(pvalue)')

            # Actually plot stuff
            top_indices_deepcombi = np.load(
                os.path.join(FINAL_RESULTS_DIR, 'deepcombi_selected_indices', '{}.npy'.format(disease_id)))
            top_indices_combi = np.load(
                os.path.join(FINAL_RESULTS_DIR, 'combi_selected_indices', '{}.npy'.format(disease_id)))

            complete_pvalues = []
            n_snps = np.zeros(22)
            for i, chromo in enumerate(chromos):

                raw_pvalues = real_pvalues(disease_id, chromo)

                if disease_id == 'CAD' and chromo != 9:
                    raw_pvalues[raw_pvalues < 1e-6] = 1

                complete_pvalues += raw_pvalues.tolist()

                n_snps[i] = len(raw_pvalues)
                offsets[i + 1] = offsets[i] + n_snps[i]
                middle_offset_history[i] = offsets[i] + int(n_snps[i] / 2)

            complete_pvalues = np.array(complete_pvalues).flatten()

            combi_selected_pvalues = np.ones(len(complete_pvalues))
            combi_selected_pvalues[top_indices_combi] = complete_pvalues[top_indices_combi]

            deepcombi_selected_pvalues = np.ones(len(complete_pvalues))
            deepcombi_selected_pvalues[top_indices_deepcombi] = complete_pvalues[top_indices_deepcombi]

            informative_idx = np.argwhere(complete_pvalues < 1e-5)

            # Color
            color = np.zeros((len(complete_pvalues), 3))
            alt = True
            for i, offset in enumerate(offsets[:-1]):
                color[offset:offsets[i + 1]] = [0, 0, 0.7] if alt else [0.4, 0.4, 0.8]
                alt = not alt
            color[informative_idx] = [0, 1, 0]

            # Plot
            raw_pvalues_ax.scatter(range(len(complete_pvalues)), -np.log10(complete_pvalues), c=color, marker='x')
            deepc_selected_pvalues_ax.scatter(range(len(complete_pvalues)), -np.log10(deepcombi_selected_pvalues),
                                              c=color, marker='x')
            c_selected_pvalues_ax.scatter(range(len(complete_pvalues)), -np.log10(combi_selected_pvalues),
                                          c=color, marker='x')

            # Set ticks
            plt.setp(raw_pvalues_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
            plt.setp(deepc_selected_pvalues_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
            plt.setp(c_selected_pvalues_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
            chrom_fig.canvas.set_window_title(disease_id)

            chrom_fig.savefig(os.path.join(FINAL_RESULTS_DIR, 'plots', 'global_manhattan.pdf'.format(disease_id)))

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
            combi_scaled_rm = np.load(os.path.join(FINAL_RESULTS_DIR, 'combi_scaled_rm', '{}.npy'.format(disease)))
            combi_scaled_rm = pd.Series(data=combi_scaled_rm[candidates_raw_pvalues_genome_peaks.index],
                                             index=candidates_raw_pvalues_genome_peaks.index)
            combined_svm_scores = pd.concat([combined_svm_scores, combi_scaled_rm])

            # DeepCombi WEIGHTS
            deepcombi_scaled_rm = np.load(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_scaled_rm', '{}.npy'.format(disease)))
            deepcombi_scaled_rm = pd.Series(
                data=deepcombi_scaled_rm[candidates_raw_pvalues_genome_peaks.index],
                index=candidates_raw_pvalues_genome_peaks.index)
            combined_deepcombi_scores = pd.concat([combined_deepcombi_scores, deepcombi_scaled_rm])

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
        plt.plot(deepcombi_fp, deepcombi_tp, label='DeepCOMBI')
        plt.plot(combi_fp, combi_tp, label='COMBI')
        plt.plot(rpvt_fp, rpvt_tp, label='RPVT')
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
        plt.plot(deepcombi_tp, deepcombi_precision, label='DeepCOMBI')
        plt.plot(combi_tp, combi_precision, label='COMBI')
        plt.plot(rpvt_tp, rpvt_precision, label='RPVT')
        plt.plot(svm_tp, svm_precision, label='SVM')
        plt.plot(dc_rm_tp, dc_rm_precision, label='DeepCOMBI weights')
        plt.xlabel('TP')
        plt.ylabel('Precision')
        plt.xlim(0, 40)

        plt.legend()
        fig.savefig(os.path.join(FINAL_RESULTS_DIR, 'plots', 'precision-tp.png'))

    def test_generate_val_acc_graph(self):
        rows_list = []
        for i, disease in enumerate(disease_IDs):
            combi_acc = np.load(os.path.join(FINAL_RESULTS_DIR, 'accuracies', disease, 'combi.npy')) * 100
            dc_acc = np.load(os.path.join(FINAL_RESULTS_DIR, 'accuracies', disease, 'deepcombi.npy')) * 100
            dc_worst_acc = np.load(
                os.path.join(ROOT_DIR, 'CROHN_C1_TRAINED_BCKP', 'accuracies', disease, 'deepcombi.npy')) * 100

            for elt in dc_worst_acc:
                rows_list.append({'Disease': disease,
                                  'Validation_Accuracy': elt,
                                  'Method': 'DeepCOMBI - Optimised on CD'})

            for elt in combi_acc:
                rows_list.append({'Disease': disease,
                                  'Validation_Accuracy': elt,
                                  'Method': 'COMBI'})
            for elt in dc_acc:
                rows_list.append({'Disease': disease,
                                  'Validation_Accuracy': elt,
                                  'Method': 'DeepCOMBI - Optimized on genome'})

        df = pd.DataFrame(rows_list)

        barplot = seaborn.barplot(x='Disease', y='Validation_Accuracy', hue='Method', data=df, capsize=.2)
        barplot.figure.savefig(os.path.join(FINAL_RESULTS_DIR, 'plots', 'combi_best_acc_comparison.png'))
