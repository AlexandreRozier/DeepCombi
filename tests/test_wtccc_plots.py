import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy
from scipy import signal

from sklearn.metrics import roc_curve, precision_recall_curve

import seaborn
import pandas as pd

from tqdm import tqdm
import numpy as np

import pdb

from parameters_complete import disease_IDs, FINAL_RESULTS_DIR, REAL_DATA_DIR, ROOT_DIR


class TestWTCCCPlots(object):

    def test_generate_per_disease_manhattan_plots(self, real_pvalues, chrom_length):
        """ Plot manhattan figures of rpvt VS deepcombi, for one specific disease
        """
        for disease_id in tqdm(disease_IDs):

            chromos = range(1, 23)
            offsets = np.zeros(len(chromos) + 1, dtype=np.int)
            middle_offset_history = np.zeros(len(chromos), dtype=np.int)

            chrom_fig, axes = plt.subplots(5, 1, sharex='col')
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

            top_indices_deepcombi = np.load(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_selected_indices', '{}.npy'.format(disease_id)))
            top_indices_combi = np.load(os.path.join(FINAL_RESULTS_DIR, 'combi_selected_indices', '{}.npy'.format(disease_id)))

            complete_pvalues = []
            n_snps = np.zeros(22)

            svm_avg_weights = np.load(os.path.join(FINAL_RESULTS_DIR, 'combi_avg_rm', '{}.npy'.format(disease_id)))
            dc_avg_weights = np.load(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_avg_rm', '{}.npy'.format(disease_id)))

            t_star_all = []
            t_star_all_combi = []
            informative_idx_deep = []
            informative_idx_combi = []
            for i, chromo in enumerate(chromos):

                raw_pvalues = real_pvalues(disease_id, chromo)

                if disease_id == 'CAD' and chromo != 9:
                    raw_pvalues[raw_pvalues < 1e-6] = 1

                complete_pvalues += raw_pvalues.tolist()

                n_snps[i] = len(raw_pvalues)
                offsets[i + 1] = offsets[i] + n_snps[i]
                middle_offset_history[i] = offsets[i] + int(n_snps[i] / 2)
                svm_avg_weights[offsets[i]:offsets[i + 1]] *= np.sqrt(chrom_length(disease_id, chromo))
                dc_avg_weights[offsets[i]:offsets[i + 1]] *= np.sqrt(chrom_length(disease_id, chromo))

                t_star_now = pd.read_pickle(os.path.join(FINAL_RESULTS_DIR,'permtest_results',disease_id, 'chrom{}-t_star.p'.format(chromo)))
                t_star_all.append(t_star_now)
                t_star_now_combi = pd.read_pickle(os.path.join(FINAL_RESULTS_DIR,'permtest_results',disease_id, 'chrom{}-t_star_combi.p'.format(chromo)))
                t_star_all_combi.append(t_star_now_combi)

                informative_idx_combi = np.append(informative_idx_combi,(raw_pvalues < t_star_now_combi))
                informative_idx_deep = np.append(informative_idx_deep,(raw_pvalues < t_star_now))

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

            c_rm_ax.scatter(range(len(complete_pvalues)), svm_avg_weights, c=color, marker='x')
            deepc_rm_ax.scatter(range(len(complete_pvalues)), dc_avg_weights, c=color, marker='x')

            color_deep = color.copy()
            color_combi = color.copy()
            
            informative_idx_deep_bool = list(map(bool,informative_idx_deep))
            informative_idx_combi_bool = list(map(bool,informative_idx_combi))

            color[informative_idx] = [0, 1, 0]
            color_deep[informative_idx_deep_bool] = [0, 1, 0]
            color_combi[informative_idx_combi_bool] = [0, 1, 0]
			
            # Plot
            raw_pvalues_ax.scatter(range(len(complete_pvalues)), -np.log10(complete_pvalues), c=color, marker='x')
            deepc_selected_pvalues_ax.scatter(range(len(complete_pvalues)), -np.log10(deepcombi_selected_pvalues), c=color_deep, marker='x')
            c_selected_pvalues_ax.scatter(range(len(complete_pvalues)), -np.log10(combi_selected_pvalues), c=color_combi, marker='x')

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
        chrom_fig.set_size_inches(18.5, 10.5)

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
        svm_scaled_weights = np.load(os.path.join(FINAL_RESULTS_DIR, 'combi_avg_rm', '{}.npy'.format(disease_id)))
        dc_scaled_weights = np.load(
            os.path.join(FINAL_RESULTS_DIR, 'deepcombi_avg_rm', '{}.npy'.format(disease_id)))

        for i, chromo in enumerate(chromos):

            raw_pvalues = real_pvalues(disease_id, chromo)

            if disease_id == 'CAD' and chromo != 9:
                raw_pvalues[raw_pvalues < 1e-6] = 1

            complete_pvalues += raw_pvalues.tolist()

            n_snps[i] = len(raw_pvalues)
            offsets[i + 1] = offsets[i] + n_snps[i]
            middle_offset_history[i] = offsets[i] + int(n_snps[i] / 2)

            svm_scaled_weights[offsets[i]:offsets[i + 1]] = svm_scaled_weights[offsets[i]:offsets[i + 1]] * np.sqrt(
                chrom_length(disease_id, i + 1))
            dc_scaled_weights[offsets[i]:offsets[i + 1]] = dc_scaled_weights[offsets[i]:offsets[i + 1]] * np.sqrt(
                chrom_length(disease_id, i + 1))

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

        # Plot
        c_rm_ax.scatter(range(len(complete_pvalues)), svm_scaled_weights, c=color, marker='x', )
        deepc_rm_ax.scatter(range(len(complete_pvalues)), dc_scaled_weights, c=color, marker='x')

        color[informative_idx] = [0, 1, 0]
        raw_pvalues_ax.scatter(range(len(complete_pvalues)), -np.log10(complete_pvalues), c=color, marker='x')


        # Set ticks
        plt.setp(raw_pvalues_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
        plt.setp(c_rm_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
        plt.setp(deepc_rm_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
        chrom_fig.canvas.set_window_title(disease_id)
        chrom_fig.tight_layout()

        chrom_fig.savefig(os.path.join(FINAL_RESULTS_DIR, 'plots', '{}-lrp.png'.format(disease_id)))

    def test_generate_global_manhattan_plots(self, real_pvalues):
        """ Plot manhattan figures of RPVT VS DeepCOMBI, for one specific disease
        """
        chrom_fig, axes = plt.subplots(7, 3, sharex='col')
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
            chrom_fig.tight_layout()

            chrom_fig.savefig(os.path.join(FINAL_RESULTS_DIR, 'plots', 'global_manhattan.pdf'.format(disease_id)))

    def test_generate_roc_recall_curves(self, real_pvalues, chrom_length):
        os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'plots'), exist_ok=True)

        combined_labels = pd.Series()
        combined_combi_pvalues = pd.DataFrame()
        combined_deepcombi_pvalues = pd.DataFrame()
        combined_rpvt_scores = pd.DataFrame()
        combined_svm_scores = pd.Series()
        combined_deepcombi_scores = pd.Series()

        for disease in tqdm(disease_IDs):
            queries = pd.read_csv(os.path.join(REAL_DATA_DIR, 'queries', '{}.txt'.format(disease)), delim_whitespace=True)

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

            pvalues_peaks_representatives = pd.DataFrame(
                data={'pvalue': raw_pvalues_genome_peaks[sorting_whole_genome_peaks_indices]},
                index=peaks_indices_genome[sorting_whole_genome_peaks_indices])

            # If the two queries match in size, add the rs_identifier field to our raw_pvalues.
            # We can assign the identifiers thanks to the ordering of the pvalues
            assert len(pvalues_peaks_representatives.index) == len(queries.index)
            pvalues_peaks_representatives['rs_identifier'] = queries.rs_identifier.tolist()

            # CREATE GROUND TRUTH LABELS
            tp_df = pd.read_csv(os.path.join(REAL_DATA_DIR, 'results_2020', '{}.txt'.format(disease)), delim_whitespace=True)
            tp_df = tp_df.rename(columns={"#SNP_A": "rs_identifier"})
            tp_df = pvalues_peaks_representatives.reset_index().merge(tp_df, on='rs_identifier', how='right').set_index('index')
            tp_df = tp_df.rename(columns={"pvalue_x": "pvalue"}).drop(columns=['pvalue_y'])

            pvalues_peak_labels = pd.Series(data=np.zeros(len(pvalues_peaks_representatives.index)), index=pvalues_peaks_representatives.index)
            pvalues_peak_labels.loc[np.intersect1d(pvalues_peaks_representatives.index, tp_df.index)] = 1
            combined_labels = pd.concat([combined_labels, pvalues_peak_labels])

            # SVM - DeepCOMBI WEIGHTS ++++++++
            combi_scaled_rm = np.load(os.path.join(FINAL_RESULTS_DIR, 'combi_scaled_rm', '{}.npy'.format(disease)))
            dc_scaled_rm = np.load(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_scaled_rm', '{}.npy'.format(disease)))
            #dc_scaled_rm = np.load(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_raw_rm', '{}.npy'.format(disease)))

            # Take chromosom length into account
            off = 0
            for i in range(22):
                l = chrom_length(disease, i + 1)
                combi_scaled_rm[off:off + l] = combi_scaled_rm[off:off + l] * np.sqrt(l)
                dc_scaled_rm[off:off + l] = dc_scaled_rm[off:off + l] * np.sqrt(l)
                off += l

            combi_scaled_rm = pd.Series(data=combi_scaled_rm[pvalues_peaks_representatives.index],  index=pvalues_peaks_representatives.index)
            combined_svm_scores = pd.concat([combined_svm_scores, combi_scaled_rm])

            dc_scaled_rm = pd.Series(data=dc_scaled_rm[pvalues_peaks_representatives.index], index=pvalues_peaks_representatives.index)
            combined_deepcombi_scores = pd.concat([combined_deepcombi_scores, dc_scaled_rm])
            # ++++

            # COMBI
            selected_combi_indices = np.load( os.path.join(FINAL_RESULTS_DIR, 'combi_selected_indices', '{}.npy'.format(disease))).flatten()
            selected_combi_pvalues = pd.Series(data=np.ones(len(pvalues_peaks_representatives.index)), index=pvalues_peaks_representatives.index)  # Build a all ones pvalues series
            idx = np.intersect1d(selected_combi_indices, pvalues_peaks_representatives.index)

            selected_combi_pvalues.loc[idx] = pvalues_peaks_representatives.loc[idx].pvalue
            assert len(selected_combi_pvalues.index) > 0
            combined_combi_pvalues = pd.concat([combined_combi_pvalues, selected_combi_pvalues])

            # DeepCombi
            selected_deepcombi_indices = np.load(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_selected_indices', '{}.npy'.format(disease))).flatten()
            selected_deepcombi_pvalues = pd.Series(data=np.ones(len(pvalues_peaks_representatives.index)), index=pvalues_peaks_representatives.index)  # Build a all ones pvalues series
            idx = np.intersect1d(selected_deepcombi_indices, pvalues_peaks_representatives.index)
            selected_deepcombi_pvalues.loc[idx] = pvalues_peaks_representatives.loc[idx].pvalue
            assert len(selected_deepcombi_pvalues.index) > 0
            combined_deepcombi_pvalues = pd.concat([combined_deepcombi_pvalues, selected_deepcombi_pvalues])

            # RPVT
            combined_rpvt_scores = pd.concat([combined_rpvt_scores, pvalues_peaks_representatives])

        # AUC stuff
        svm_fpr, svm_tpr, _ = roc_curve(combined_labels.values,combined_svm_scores.values)
        svm_tp = svm_tpr * (combined_labels.values == 1).sum()
        svm_fp = svm_fpr * (combined_labels.values != 1).sum()

        dc_rm_fpr, dc_rm_tpr, _ = roc_curve(combined_labels.values,combined_deepcombi_scores.values)
        dc_rm_tp = dc_rm_tpr * (combined_labels.values == 1).sum()
        dc_rm_fp = dc_rm_fpr * (combined_labels.values != 1).sum()

        combi_fpr, combi_tpr, _ = roc_curve(combined_labels.values, -np.log10(combined_combi_pvalues.values))
        combi_tp = combi_tpr * (combined_labels.values == 1).sum()
        combi_fp = combi_fpr * (combined_labels.values != 1).sum()

        deepcombi_fpr, deepcombi_tpr, _ = roc_curve(combined_labels.values, -np.log10(combined_deepcombi_pvalues.values))
        deepcombi_tp = deepcombi_tpr * (combined_labels.values == 1).sum()
        deepcombi_fp = deepcombi_fpr * (combined_labels.values != 1).sum()


        rpvt_fpr, rpvt_tpr, _ = roc_curve(combined_labels.values, -np.log10(combined_rpvt_scores.pvalue.values))
        rpvt_tp = rpvt_tpr * (combined_labels.values == 1).sum()
        rpvt_fp = rpvt_fpr * (combined_labels.values != 1).sum()

        # CURVES must stop somewhere
        combi_fp = combi_fp[combi_fp < 80]
        combi_tp = combi_tp[:len(combi_fp)]
        deepcombi_fp = deepcombi_fp[deepcombi_fp < 80]
        deepcombi_tp = deepcombi_tp[:len(deepcombi_fp)]

        fig = plt.figure(figsize=[15,9])
        plt.subplot(121)
        plt.plot(rpvt_fp, rpvt_tp, label='RPVT', color='lightsteelblue', linewidth=2)
        plt.plot(combi_fp, combi_tp, label='COMBI', color='darkblue', linewidth=2)
        plt.plot(deepcombi_fp, deepcombi_tp, label='DeepCOMBI', color='fuchsia', linewidth=3)
        plt.plot(svm_fp, svm_tp, label='SVM weights', linestyle='--', color='darkblue', linewidth=2)
        plt.plot(dc_rm_fp, dc_rm_tp, label='LRP scores', linestyle='--', color='fuchsia', linewidth=2)
        plt.xlabel('Number of false positives', fontsize=14)
        plt.ylabel('Number of true positives', fontsize=14)
        plt.xlim(0, 40)
        plt.ylim(0, 40)
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(loc='lower right', fontsize=14)
        #fig.savefig(os.path.join(FINAL_RESULTS_DIR, 'plots', 'auc.png'))

        # Precision recall stuff

        combi_precision, combi_recall, thresholds = precision_recall_curve(combined_labels.values, -np.log10(combined_combi_pvalues.values))
        combi_tp = combi_recall * (combined_labels.values == 1).sum()

        deepcombi_precision, deepcombi_recall, _ = precision_recall_curve(combined_labels.values, -np.log10(combined_deepcombi_pvalues.values))
        deepcombi_tp = deepcombi_recall * (combined_labels.values == 1).sum()

        rpvt_precision, rpvt_recall, _ = precision_recall_curve(combined_labels.values, -np.log10(combined_rpvt_scores.pvalue.values))
        rpvt_tp = rpvt_recall * (combined_labels.values == 1).sum()

        svm_precision, svm_recall, _ = precision_recall_curve(combined_labels.values, combined_svm_scores.values)
        svm_tp = svm_recall * (combined_labels.values == 1).sum()

        dc_rm_precision, dc_rm_recall, _ = precision_recall_curve(combined_labels.values, combined_deepcombi_scores.values)
        dc_rm_tp = dc_rm_recall * (combined_labels.values == 1).sum()

        combi_tp = combi_tp[combi_tp < 70]
        combi_precision = combi_precision[-len(combi_tp):]
        deepcombi_tp = deepcombi_tp[deepcombi_tp < 70]
        deepcombi_precision = deepcombi_precision[-len(deepcombi_tp):]

        #fig = plt.figure()
        plt.subplot(122)
        plt.plot(rpvt_tp, rpvt_precision, label='RPVT', color='lightsteelblue', linewidth=2)
        plt.plot(combi_tp, combi_precision, label='COMBI', color='darkblue', linewidth=2)
        plt.plot(deepcombi_tp, deepcombi_precision, label='DeepCOMBI', color='fuchsia', linewidth=3)
        plt.plot(svm_tp, svm_precision, label='SVM weights', linestyle='--', color='darkblue', linewidth=2)
        plt.plot(dc_rm_tp, dc_rm_precision, label='LRP scores', linestyle='--', color='fuchsia', linewidth=2)
        plt.xlabel('Number of true positives', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.xlim(0, 40)
        plt.ylim(0,1.0)

        plt.legend(loc='lower right', fontsize=14)

        fig.savefig(os.path.join(FINAL_RESULTS_DIR, 'plots', 'all_curves_final_NAR.png'), bbox_inches='tight')

    def test_generate_val_acc_graph(self):
        rows_list = []
        for i, disease in enumerate(disease_IDs):
            combi_acc = np.load(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease, 'combi_prec.npy')) * 100
            dc_acc = np.load(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease, 'deepcombi_prec.npy')) * 100
            #dc_worst_acc = np.load(
            #    os.path.join(ROOT_DIR, 'CROHN_C1_TRAINED_BCKP', 'accuracies', disease, 'deepcombi.npy')) * 100

            #for elt in dc_worst_acc:
            #    rows_list.append({'Disease': disease,
            #                      'Validation_Accuracy': elt,
            #                      'Method': 'DeepCOMBI - Optimised on CD'})

            for elt in combi_acc:
                rows_list.append({'Disease': disease,
                                  'Validation_Accuracy': elt,
                                  'Method': 'COMBI'})
            for elt in dc_acc:
                rows_list.append({'Disease': disease,
                                  'Validation_Accuracy': elt,
                                  'Method': 'DeepCOMBI'})

        df = pd.DataFrame(rows_list)

        barplot = seaborn.barplot(x='Disease', y='Validation_Accuracy', hue='Method', data=df, capsize=.2)
        barplot.figure.savefig(os.path.join(FINAL_RESULTS_DIR, 'plots', 'val_prec_comparison.png'))


    def test_permtestresults(self, real_pvalues, chrom_length):
        os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'permtest_results'), exist_ok=True)

        # For each disease write files with significant RPVT, COMBI and DeepCOMBI SNPs and create table of performance with thresholds 
		
		# From manhattan script:
		# pvalues of all methods: complete_pvalues, combi_selected_pvalues, deepcombi_selected_pvalues
		# Get indices of significant SNPs according to t_stars of all methods: informative_idx, informative_idx_combi, informative_idx_deep
		
        counter_pos_rpvt = 0
        counter_neg_rpvt = 0
        counter_pos_combi = 0
        counter_neg_combi = 0
        counter_pos_deep = 0
        counter_neg_deep = 0

        for disease_id in tqdm(disease_IDs):

            top_indices_deepcombi = np.load(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_selected_indices', '{}.npy'.format(disease_id)))
            top_indices_combi = np.load(os.path.join(FINAL_RESULTS_DIR, 'combi_selected_indices', '{}.npy'.format(disease_id)))


            # From curve script:
            # Get identifiers of all SNPs - pvalues_peaks_representatives
            # Get label of all SNPs - pvalues_peak_labels

            queries = pd.read_csv(os.path.join(REAL_DATA_DIR, 'queries', '{}.txt'.format(disease_id)), delim_whitespace=True)

            # Preselects indices of interest (at PEAKS, on pvalues smaller than 1e-4)
            offset = 0
            peaks_indices_genome = []
            raw_pvalues_genome = []

            for chromo in tqdm(range(1, 23)):
                pvalues = real_pvalues(disease_id, chromo)

                pvalues_104 = np.ones(pvalues.shape)
                pvalues_104[pvalues < 1e-4] = pvalues[pvalues < 1e-4]
                peaks_indices, _ = scipy.signal.find_peaks(-np.log10(pvalues_104), distance=150)
                peaks_indices += offset

                # BUGFIX for CAD
                if disease_id == 'CAD' and chromo != 9:
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

            pvalues_peaks_representatives = pd.DataFrame(
                data={'pvalue': raw_pvalues_genome_peaks[sorting_whole_genome_peaks_indices]},
                index=peaks_indices_genome[sorting_whole_genome_peaks_indices])

            # If the two queries match in size, add the rs_identifier field to our raw_pvalues.
            # We can assign the identifiers thanks to the ordering of the pvalues
            assert len(pvalues_peaks_representatives.index) == len(queries.index)
            pvalues_peaks_representatives['rs_identifier'] = queries.rs_identifier.tolist()

            # CREATE GROUND TRUTH LABELS
            tp_df = pd.read_csv(os.path.join(REAL_DATA_DIR, 'results_2020', '{}.txt'.format(disease_id)), delim_whitespace=True)
            tp_df = tp_df.rename(columns={"#SNP_A": "rs_identifier"})
            tp_df = pvalues_peaks_representatives.reset_index().merge(tp_df,
                                                                      on='rs_identifier',
                                                                      how='right').set_index('index')
            tp_df = tp_df.rename(columns={"pvalue_x": "pvalue"}).drop(columns=['pvalue_y'])

            pvalues_peak_labels = pd.Series(data=np.zeros(len(pvalues_peaks_representatives.index)), index=pvalues_peaks_representatives.index)
            pvalues_peak_labels.loc[np.intersect1d(pvalues_peaks_representatives.index, tp_df.index)] = 1

            # Get identifiers of all SNPs - pvalues_peaks_representatives
            # Get label of all SNPs - pvalues_peak_labels
            # pvalues of all methods: complete_pvalues, combi_selected_pvalues, deepcombi_selected_pvalues
            # Get indices of significant SNPs according to t_stars of all methods: informative_idx, informative_idx_combi, informative_idx_deep
			
            # representatives of RPVT

            # Preselects indices of interest (at PEAKS, on pvalues smaller than 10-5)
            offset = 0
            peaks_indices_genome_rpvt = []
            raw_pvalues_genome_rpvt = []

            for chromo in tqdm(range(1, 23)):
                raw_pvalues = real_pvalues(disease_id, chromo)
                #if disease_id == 'CAD' and chromo == 6:
                #    pdb.set_trace()				

                if disease_id == 'CAD' and chromo != 9:
                    raw_pvalues[raw_pvalues < 1e-6] = 1

                pvalues = np.ones(raw_pvalues.shape)
                pvalues[raw_pvalues < 1e-5]=raw_pvalues[raw_pvalues < 1e-5]

                peaks_indices, _ = scipy.signal.find_peaks(-np.log10(pvalues), distance=150)
                peaks_indices += offset

                offset += len(pvalues)
                raw_pvalues_genome_rpvt += pvalues.tolist()
                peaks_indices_genome_rpvt += peaks_indices.tolist()
                del pvalues, raw_pvalues

            raw_pvalues_genome_rpvt = np.array(raw_pvalues_genome_rpvt)
            peaks_indices_genome_rpvt = np.array(peaks_indices_genome_rpvt)

            # pvalues being < 1e-5 and belonging to a peak
            raw_pvalues_genome_peaks_rpvt = raw_pvalues_genome_rpvt[peaks_indices_genome_rpvt]
            sorting_whole_genome_peaks_indices_rpvt = np.argsort(raw_pvalues_genome_peaks_rpvt)

            pvalues_peaks_representatives_rpvt = pd.DataFrame(data={'pvalue': raw_pvalues_genome_peaks_rpvt[sorting_whole_genome_peaks_indices_rpvt]}, index=peaks_indices_genome_rpvt[sorting_whole_genome_peaks_indices_rpvt])

            results_rpvt = pd.DataFrame(columns=['chromosome','rs_identifier','pvalue'])
            labels_rpvt=np.zeros(len(pvalues_peaks_representatives_rpvt))

            for i in range(len(pvalues_peaks_representatives_rpvt)):
                reps_index = (np.abs(pvalues_peaks_representatives.index - pvalues_peaks_representatives_rpvt.index[i])).argmin()
                queries_index = np.where(pvalues_peaks_representatives.rs_identifier.iloc[reps_index]==queries.rs_identifier)[0][0]
                results_rpvt.loc[i] = queries.iloc[queries_index]
                results_rpvt['pvalue'][i] =  pvalues_peaks_representatives_rpvt.pvalue.iloc[i]
                labels_rpvt[i]=pvalues_peak_labels.data[queries_index]
                if queries_index.size==0:
                    print('RS Identifier not in queries!')
                    pdb.set_trace()

            results_rpvt['evaluation_label'] = labels_rpvt
			
            counter_pos_rpvt += np.sum(labels_rpvt)
            counter_neg_rpvt += (len(labels_rpvt)-np.sum(labels_rpvt))

            results_rpvt.to_csv(os.path.join(FINAL_RESULTS_DIR, 'permtest_results',disease_id, 'significant_RPVT'), sep='\t')

            # representatives of COMBI

            # Preselects indices of interest (at PEAKS, on pvalues smaller than t_star_combi)
            offset = 0
            peaks_indices_genome_combi = []
            raw_pvalues_genome_combi = []

            for chromo in tqdm(range(1, 23)):
                raw_pvalues = real_pvalues(disease_id, chromo)

                if disease_id == 'CAD' and chromo != 9:
                    raw_pvalues[raw_pvalues < 1e-6] = 1

                # Only COMBI selected pvalues 
                top_indices_combi_now = top_indices_combi[(top_indices_combi<offset+len(raw_pvalues))&(top_indices_combi>=offset)]-offset

                pvalues_combi = np.ones(raw_pvalues.shape)
                pvalues_combi[top_indices_combi_now] = raw_pvalues[top_indices_combi_now]

                t_star_now_combi = pd.read_pickle(os.path.join(FINAL_RESULTS_DIR,'permtest_results',disease_id, 'chrom{}-t_star_combi.p'.format(chromo)))

                pvalues = np.ones(pvalues_combi.shape)
                pvalues[pvalues_combi < t_star_now_combi]=pvalues_combi[pvalues_combi < t_star_now_combi]

                #if t_star_now_combi > 1e-4:
                #    pdb.set_trace()
                peaks_indices, _ = scipy.signal.find_peaks(-np.log10(pvalues), distance=150)
                peaks_indices += offset

                ## BUGFIX for CAD
                #if disease_id == 'CAD' and chromo != 9:
                #    pvalues[pvalues < 1e-6] = 1

                offset += len(pvalues)
                raw_pvalues_genome_combi += pvalues.tolist()
                peaks_indices_genome_combi += peaks_indices.tolist()
                del pvalues, raw_pvalues

            raw_pvalues_genome_combi = np.array(raw_pvalues_genome_combi)
            peaks_indices_genome_combi = np.array(peaks_indices_genome_combi)

            # pvalues being < t_star_combi and belonging to a peak
            raw_pvalues_genome_peaks_combi = raw_pvalues_genome_combi[peaks_indices_genome_combi]
            sorting_whole_genome_peaks_indices_combi = np.argsort(raw_pvalues_genome_peaks_combi)

            pvalues_peaks_representatives_combi = pd.DataFrame(data={'pvalue': raw_pvalues_genome_peaks_combi[sorting_whole_genome_peaks_indices_combi]}, index=peaks_indices_genome_combi[sorting_whole_genome_peaks_indices_combi])
            results_combi = pd.DataFrame(columns=['chromosome','rs_identifier','pvalue'])
            labels_combi=np.zeros(len(pvalues_peaks_representatives_combi))

            for i in range(len(pvalues_peaks_representatives_combi)):
                reps_index = (np.abs(pvalues_peaks_representatives.index - pvalues_peaks_representatives_combi.index[i])).argmin()
                queries_index = np.where(pvalues_peaks_representatives.rs_identifier.iloc[reps_index]==queries.rs_identifier)[0][0]
                results_combi.loc[i] = queries.iloc[queries_index]
                results_combi['pvalue'][i] =  pvalues_peaks_representatives_combi.pvalue.iloc[i]
                labels_combi[i]=pvalues_peak_labels.data[queries_index]
                if queries_index.size==0:
                    print('RS Identifier not in queries!')
                    pdb.set_trace()

            results_combi['evaluation_label'] = labels_combi
			
            counter_pos_combi += np.sum(labels_combi)
            counter_neg_combi += (len(labels_combi)-np.sum(labels_combi))

            results_combi.to_csv(os.path.join(FINAL_RESULTS_DIR, 'permtest_results',disease_id, 'significant_COMBI'), sep='\t')			


            # representatives of DeepCOMBI

            # Preselects indices of interest (at PEAKS, on pvalues smaller than t_star)
            offset = 0
            peaks_indices_genome_deep = []
            raw_pvalues_genome_deep = []

            for chromo in tqdm(range(1, 23)):
                raw_pvalues = real_pvalues(disease_id, chromo)

                if disease_id == 'CAD' and chromo != 9:
                    raw_pvalues[raw_pvalues < 1e-6] = 1

                # Only DeepCOMBI selected pvalues 
                top_indices_deep_now = top_indices_deepcombi[(top_indices_deepcombi<offset+len(raw_pvalues))&(top_indices_deepcombi>=offset)]-offset

                pvalues_deep = np.ones(raw_pvalues.shape)
                pvalues_deep[top_indices_deep_now] = raw_pvalues[top_indices_deep_now]
				
                t_star_now_deep = pd.read_pickle(os.path.join(FINAL_RESULTS_DIR,'permtest_results',disease_id, 'chrom{}-t_star.p'.format(chromo)))

                pvalues = np.ones(pvalues_deep.shape)
                pvalues[pvalues_deep < t_star_now_deep]=pvalues_deep[pvalues_deep < t_star_now_deep]

                #if t_star_now_deep > 1e-4:
                #    pdb.set_trace()
                peaks_indices, _ = scipy.signal.find_peaks(-np.log10(pvalues), distance=150)
                peaks_indices += offset

                ## BUGFIX for CAD
                #if disease_id == 'CAD' and chromo != 9:
                #    pvalues[pvalues < 1e-6] = 1

                offset += len(pvalues)
                raw_pvalues_genome_deep += pvalues.tolist()
                peaks_indices_genome_deep += peaks_indices.tolist()
                del pvalues, raw_pvalues

            raw_pvalues_genome_deep = np.array(raw_pvalues_genome_deep)
            peaks_indices_genome_deep = np.array(peaks_indices_genome_deep)

            # pvalues being < t_star_deep and belonging to a peak
            raw_pvalues_genome_peaks_deep = raw_pvalues_genome_deep[peaks_indices_genome_deep]
            sorting_whole_genome_peaks_indices_deep = np.argsort(raw_pvalues_genome_peaks_deep)

            pvalues_peaks_representatives_deep = pd.DataFrame(data={'pvalue': raw_pvalues_genome_peaks_deep[sorting_whole_genome_peaks_indices_deep]}, index=peaks_indices_genome_deep[sorting_whole_genome_peaks_indices_deep])

            results_deep = pd.DataFrame(columns=['chromosome','rs_identifier','pvalue'])
            labels_deep=np.zeros(len(pvalues_peaks_representatives_deep))

            for i in range(len(pvalues_peaks_representatives_deep)):
                reps_index = (np.abs(pvalues_peaks_representatives.index - pvalues_peaks_representatives_deep.index[i])).argmin()
                queries_index = np.where(pvalues_peaks_representatives.rs_identifier.iloc[reps_index]==queries.rs_identifier)[0][0]
                results_deep.loc[i] = queries.iloc[queries_index]
                results_deep['pvalue'][i] =  pvalues_peaks_representatives_deep.pvalue.iloc[i]
                labels_deep[i]=pvalues_peak_labels.data[queries_index]
                if queries_index.size==0:
                    print('RS Identifier not in queries!')
                    pdb.set_trace()
            results_deep['evaluation_label'] = labels_deep
		
            counter_pos_deep += np.sum(labels_deep)
            counter_neg_deep += (len(labels_deep)-np.sum(labels_deep))
			
            results_deep.to_csv(os.path.join(FINAL_RESULTS_DIR, 'permtest_results',disease_id, 'significant_deepCOMBI'), sep='\t')
        print([counter_pos_rpvt, counter_pos_combi, counter_pos_deep])
        print([counter_neg_rpvt, counter_neg_combi, counter_neg_deep])
		

    def test_generate_all_diseases_manhattan_plots(self, real_pvalues, chrom_length):
        """ Plot manhattan figures of rpvt VS deepcombi, for one specific disease
        """
		
        chrom_fig, axes = plt.subplots(7, 3, sharex='col')
        plt.setp(axes, ylim=(0,15))

        chrom_fig.set_size_inches(33, 22)

        bd_ax = axes[0,0]
        bd_combi_ax = axes[0,1]
        bd_deepcombi_ax = axes[0,2]

        cad_ax = axes[1,0]
        cad_combi_ax = axes[1,1]
        cad_deepcombi_ax = axes[1,2]

        cd_ax = axes[2,0]
        cd_combi_ax = axes[2,1]
        cd_deepcombi_ax = axes[2,2]

        ht_ax = axes[3,0]
        ht_combi_ax = axes[3,1]
        ht_deepcombi_ax = axes[3,2]
		
        ra_ax = axes[4,0]
        ra_combi_ax = axes[4,1]
        ra_deepcombi_ax = axes[4,2]

        t1d_ax = axes[5,0]
        t1d_combi_ax = axes[5,1]
        t1d_deepcombi_ax = axes[5,2]

        t2d_ax = axes[6,0]
        t2d_combi_ax = axes[6,1]
        t2d_deepcombi_ax = axes[6,2]
		

		# Set title
        bd_ax.set_title('RPVT', fontsize=22)
        bd_combi_ax.set_title('COMBI method', fontsize=22)
        bd_deepcombi_ax.set_title('DeepCOMBI method', fontsize=22)

        bd_ax.set_ylabel('Bipolar disorder', fontsize=20, rotation=0, ha='right')
        ht_ax.set_ylabel('Hypertension', fontsize=20, rotation=0, ha='right')
        cad_ax.set_ylabel('Coronary artery disease', fontsize=20, rotation=0, ha='right')
        cd_ax.set_ylabel("Crohn's disease", fontsize=20, rotation=0, ha='right')
        ra_ax.set_ylabel("Rheumatoid arthritis", fontsize=20, rotation=0, ha='right')
        t1d_ax.set_ylabel("Type 1 diabetes", fontsize=20, rotation=0, ha='right')
        t2d_ax.set_ylabel("Type 2 diabetes", fontsize=20, rotation=0, ha='right')
		
        t2d_ax.set_xlabel('Chromosome', fontsize=18)
        t2d_combi_ax.set_xlabel('Chromosome', fontsize=18)
        t2d_deepcombi_ax.set_xlabel('Chromosome', fontsize=18)

		# Set labels
		
        #ht_ax.set_ylabel('-log_10(pvalue)')
        ht_combi_ax.set_ylabel('$-log_{10}$(p-value)', fontsize=18)
        ht_deepcombi_ax.set_ylabel('$-log_{10}$(p-value)', fontsize=18)
        #cd_ax.set_ylabel('SVM weights in %')
        #ra_ax.set_ylabel('LRP relevance mapping in %')
        
        counter=0
        for disease_id in tqdm(disease_IDs):

            chromos = range(1, 23)
            offsets = np.zeros(len(chromos) + 1, dtype=np.int)
            middle_offset_history = np.zeros(len(chromos), dtype=np.int)

            top_indices_deepcombi = np.load(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_selected_indices', '{}.npy'.format(disease_id)))
            top_indices_combi = np.load(os.path.join(FINAL_RESULTS_DIR, 'combi_selected_indices', '{}.npy'.format(disease_id)))

            complete_pvalues = []
            n_snps = np.zeros(22)

            svm_avg_weights = np.load(os.path.join(FINAL_RESULTS_DIR, 'combi_avg_rm', '{}.npy'.format(disease_id)))
            dc_avg_weights = np.load(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_avg_rm', '{}.npy'.format(disease_id)))

            t_star_all = []
            t_star_all_combi = []
            informative_idx_deep = []
            informative_idx_combi = []
            for i, chromo in enumerate(chromos):

                raw_pvalues = real_pvalues(disease_id, chromo)

                if disease_id == 'CAD' and chromo != 9:
                    raw_pvalues[raw_pvalues < 1e-6] = 1

                complete_pvalues += raw_pvalues.tolist()

                n_snps[i] = len(raw_pvalues)
                offsets[i + 1] = offsets[i] + n_snps[i]
                middle_offset_history[i] = offsets[i] + int(n_snps[i] / 2)
                svm_avg_weights[offsets[i]:offsets[i + 1]] *= np.sqrt(chrom_length(disease_id, chromo))
                dc_avg_weights[offsets[i]:offsets[i + 1]] *= np.sqrt(chrom_length(disease_id, chromo))

                t_star_now = pd.read_pickle(os.path.join(FINAL_RESULTS_DIR,'permtest_results',disease_id, 'chrom{}-t_star.p'.format(chromo)))
                t_star_all.append(t_star_now)
                t_star_now_combi = pd.read_pickle(os.path.join(FINAL_RESULTS_DIR,'permtest_results',disease_id, 'chrom{}-t_star_combi.p'.format(chromo)))
                t_star_all_combi.append(t_star_now_combi)

                informative_idx_combi = np.append(informative_idx_combi,(raw_pvalues < t_star_now_combi))
                informative_idx_deep = np.append(informative_idx_deep,(raw_pvalues < t_star_now))
				
            for i, chromo in enumerate(chromos):
                axes[counter, 0].axhline(y=5, xmin=offsets[i]/offsets[22], xmax=offsets[i+1]/offsets[22], linestyle='--', color='black')
                axes[counter, 1].axhline(y=-np.log10(t_star_all_combi[i]), xmin=offsets[i]/offsets[22], xmax=offsets[i+1]/offsets[22], linestyle='--', color='black')
                axes[counter, 2].axhline(y=-np.log10(t_star_all[i]), xmin=offsets[i]/offsets[22], xmax=offsets[i+1]/offsets[22], linestyle='--', color='black')

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

            #cd_ax.scatter(range(len(complete_pvalues)), svm_avg_weights, c=color, marker='x')
            #ra_ax.scatter(range(len(complete_pvalues)), dc_avg_weights, c=color, marker='x')

            color_deep = color.copy()
            color_combi = color.copy()
            
            informative_idx_deep_bool = list(map(bool,informative_idx_deep))
            informative_idx_combi_bool = list(map(bool,informative_idx_combi))

            color[informative_idx] = [0, 1, 0]
            color_deep[informative_idx_deep_bool] = [0, 1, 0]
            color_combi[informative_idx_combi_bool] = [0, 1, 0]
			
            # Plot
            axes[counter, 0].scatter(range(len(complete_pvalues)), -np.log10(complete_pvalues), c=color, marker='.')
            axes[counter, 1].scatter(range(len(complete_pvalues)), -np.log10(combi_selected_pvalues), c=color_combi, marker='.')
            axes[counter, 2].scatter(range(len(complete_pvalues)), -np.log10(deepcombi_selected_pvalues), c=color_deep, marker='.')

            # Set ticks
            plt.setp(axes[counter, 0], xticks=middle_offset_history, xticklabels=range(1, 23))
            plt.setp(axes[counter, 0].get_xticklabels(), fontsize=16)
            plt.setp(axes[counter, 1], xticks=middle_offset_history, xticklabels=range(1, 23))
            plt.setp(axes[counter, 1].get_xticklabels(), fontsize=16)
            plt.setp(axes[counter, 2], xticks=middle_offset_history, xticklabels=range(1, 23))
            plt.setp(axes[counter, 2].get_xticklabels(), fontsize=16)
            #plt.setp(cd_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
            #plt.setp(ra_ax, xticks=middle_offset_history, xticklabels=range(1, 23))
            #chrom_fig.canvas.set_window_title(disease_id)

            counter += 1
        plt.setp(axes, xlim=(0,offsets[22]))

        plt.setp(bd_ax.get_yticklabels(), fontsize=16)
        plt.setp(bd_combi_ax.get_yticklabels(), fontsize=16)
        plt.setp(bd_deepcombi_ax.get_yticklabels(), fontsize=16)

        plt.setp(cad_ax.get_yticklabels(), fontsize=16)
        plt.setp(cad_combi_ax.get_yticklabels(), fontsize=16)
        plt.setp(cad_deepcombi_ax.get_yticklabels(), fontsize=16)

        plt.setp(cd_ax.get_yticklabels(), fontsize=16)
        plt.setp(cd_combi_ax.get_yticklabels(), fontsize=16)
        plt.setp(cd_deepcombi_ax.get_yticklabels(), fontsize=16)

        plt.setp(ht_ax.get_yticklabels(), fontsize=16)
        plt.setp(ht_combi_ax.get_yticklabels(), fontsize=16)
        plt.setp(ht_deepcombi_ax.get_yticklabels(), fontsize=16)

        plt.setp(ra_ax.get_yticklabels(), fontsize=16)
        plt.setp(ra_combi_ax.get_yticklabels(), fontsize=16)
        plt.setp(ra_deepcombi_ax.get_yticklabels(), fontsize=16)

        plt.setp(t1d_ax.get_yticklabels(), fontsize=16)
        plt.setp(t1d_combi_ax.get_yticklabels(), fontsize=16)
        plt.setp(t1d_deepcombi_ax.get_yticklabels(), fontsize=16)

        plt.setp(t2d_ax.get_yticklabels(), fontsize=16)
        plt.setp(t2d_combi_ax.get_yticklabels(), fontsize=16)
        plt.setp(t2d_deepcombi_ax.get_yticklabels(), fontsize=16)


        chrom_fig.savefig(os.path.join(FINAL_RESULTS_DIR, 'plots', 'all-diseases-manhattan-final-NAR.png'), bbox_inches='tight')


    def test_generate_all_val_accs(self):
        rows_list = []

        combi_acc_all_diseases = []
        dc_acc_all_diseases = []
        for i, disease in enumerate(disease_IDs):
            combi_acc = np.load(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease, 'combi_acc.npy')) * 100
            dc_acc = np.load(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease, 'deepcombi_acc.npy')) * 100
            combi_acc_all_diseases=np.concatenate((combi_acc_all_diseases,combi_acc))
            dc_acc_all_diseases=np.concatenate((dc_acc_all_diseases, dc_acc))
        for elt in combi_acc_all_diseases:
            rows_list.append({'Validation measure': 'Accuracy', 'Validation_Accuracy': elt, 'Method': 'SVM (COMBI)'})
        for elt in dc_acc_all_diseases:
            rows_list.append({'Validation measure': 'Accuracy', 'Validation_Accuracy': elt,  'Method': 'DNN (DeepCOMBI)'})

        combi_acc_all_diseases = []
        dc_acc_all_diseases = []
        for i, disease in enumerate(disease_IDs):
            combi_acc = np.load(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease, 'combi_balacc.npy')) * 100
            dc_acc = np.load(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease, 'deepcombi_balacc.npy')) * 100
            combi_acc_all_diseases=np.concatenate((combi_acc_all_diseases,combi_acc))
            dc_acc_all_diseases=np.concatenate((dc_acc_all_diseases, dc_acc))
        for elt in combi_acc_all_diseases:
            rows_list.append({'Validation measure': 'Balanced accuracy', 'Validation_Accuracy': elt, 'Method': 'SVM (COMBI)'})
        for elt in dc_acc_all_diseases:
            rows_list.append({'Validation measure': 'Balanced accuracy', 'Validation_Accuracy': elt,  'Method': 'DNN (DeepCOMBI)'})

        combi_acc_all_diseases = []
        dc_acc_all_diseases = []
        for i, disease in enumerate(disease_IDs):
            combi_acc = np.load(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease, 'combi_auc.npy')) * 100
            dc_acc = np.load(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease, 'deepcombi_auc.npy')) * 100
            combi_acc_all_diseases=np.concatenate((combi_acc_all_diseases,combi_acc))
            dc_acc_all_diseases=np.concatenate((dc_acc_all_diseases, dc_acc))
        for elt in combi_acc_all_diseases:
            rows_list.append({'Validation measure': 'AUC ROC', 'Validation_Accuracy': elt, 'Method': 'SVM (COMBI)'})
        for elt in dc_acc_all_diseases:
            rows_list.append({'Validation measure': 'AUC ROC', 'Validation_Accuracy': elt,  'Method': 'DNN (DeepCOMBI)'})
			
        combi_acc_all_diseases = []
        dc_acc_all_diseases = []
        for i, disease in enumerate(disease_IDs):
            combi_acc = np.load(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease, 'combi_prec.npy')) * 100
            dc_acc = np.load(os.path.join(FINAL_RESULTS_DIR, 'valaccs', disease, 'deepcombi_prec.npy')) * 100
            combi_acc_all_diseases=np.concatenate((combi_acc_all_diseases,combi_acc))
            dc_acc_all_diseases=np.concatenate((dc_acc_all_diseases, dc_acc))
        for elt in combi_acc_all_diseases:
            rows_list.append({'Validation measure': 'AUC precision recall', 'Validation_Accuracy': elt, 'Method': 'SVM (COMBI)'})
        for elt in dc_acc_all_diseases:
            rows_list.append({'Validation measure': 'AUC precision recall', 'Validation_Accuracy': elt,  'Method': 'DNN (DeepCOMBI)'})
			
        df = pd.DataFrame(rows_list)

        barplot=seaborn.barplot(x='Validation measure', y='Validation_Accuracy', hue='Method', data=df, capsize=.2, palette='gnuplot2')		
        barplot.set_xlabel('Validation measure', fontsize=14)
        barplot.set_ylabel('Value in %', fontsize=14)
        plt.legend(loc= 'lower right')

        barplot.figure.savefig(os.path.join(FINAL_RESULTS_DIR, 'plots', 'all_vals_comparison_NAR.png'), bbox_inches='tight')
