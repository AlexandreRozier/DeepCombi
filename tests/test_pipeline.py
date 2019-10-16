import os

import matplotlib

matplotlib.use('Qt5Agg')

import numpy as np
import pandas as pd
from keras.models import load_model
from tqdm import tqdm

from combi import deepcombi_method, combi_method, real_classifier
from helpers import char_matrix_to_featmat, postprocess_weights_without_avg, postprocess_weights, moving_average
from parameters_complete import real_pnorm_feature_scaling, real_p_pnorm_filter, \
    p_svm, real_top_k, FINAL_RESULTS_DIR, filter_window_size, disease_IDs


class TestPipeline(object):

    def test_save_deepcombi_accuracies(self):
        for disease in tqdm(disease_IDs):
            scores = []
            for chrom in tqdm(range(1, 23)):
                data = pd.read_csv(os.path.join(FINAL_RESULTS_DIR, 'csv_logs', disease, str(chrom)))
                scores += data.tail(1)['val_acc'].values.tolist()
            np.save(os.path.join(FINAL_RESULTS_DIR, 'accuracies', disease, 'deepcombi'), scores)

    def test_save_combi_scores_rm_and_indices(self, real_h5py_data, real_labels, chrom_length):

        """ Extract indices gotten from combi
        """
        disease = disease_IDs[int(os.environ['SGE_TASK_ID']) - 1]
        offset = 0
        selected_indices = []
        total_raw_rm = np.empty((0, 3))
        labels = real_labels(disease)
        
        for chromo in tqdm(range(1, 23)):
            data = real_h5py_data(disease, chromo)
            fm_2D = char_matrix_to_featmat(data, '2d', real_pnorm_feature_scaling)
            # Save weights + indices
            selected_idx, _, raw_rm = combi_method(real_classifier, data, fm_2D,
                                                   labels, filter_window_size, real_p_pnorm_filter,
                                                   p_svm, top_k=real_top_k)
            # Save validation accuracies

            indices = offset + selected_idx

            selected_indices += indices.tolist()
            offset += chrom_length(disease,chromo)
            total_raw_rm = np.append(total_raw_rm, raw_rm, axis=0)

            del data, fm_2D

        os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'combi_selected_indices'), exist_ok=True)
        os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'combi_raw_rm'), exist_ok=True)

        np.save(os.path.join(FINAL_RESULTS_DIR, 'combi_selected_indices', disease), selected_indices)
        np.save(os.path.join(FINAL_RESULTS_DIR, 'combi_raw_rm', disease), total_raw_rm)

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


    def test_save_scaled_averaged_rm(self, chrom_length):

        os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'combi_avg_rm'), exist_ok=True)
        os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'combi_scaled_rm'), exist_ok=True)
        os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_avg_rm'), exist_ok=True)
        os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_scaled_rm'), exist_ok=True)

        for disease in disease_IDs:
            combi_raw_rm = np.load(os.path.join(FINAL_RESULTS_DIR, 'combi_raw_rm', '{}.npy'.format(disease)))
            combi_scaled_rm = np.zeros(combi_raw_rm.shape[0])
            combi_avg_rm = np.zeros(combi_raw_rm.shape[0])

            deepcombi_raw_rm = np.load(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_raw_rm','{}.npy'.format(disease)))
            deepcombi_scaled_rm = np.zeros(deepcombi_raw_rm.shape[0])
            deepcombi_avg_rm = np.zeros(deepcombi_raw_rm.shape[0])
            offset = 0

            assert not np.isnan(combi_raw_rm.sum())
            assert not np.isnan(deepcombi_raw_rm.sum())
            assert np.absolute(deepcombi_raw_rm).sum() > 0 # Check for 0 only relevance mappings

            for chrom in range(1, 23):
                chromo_length = chrom_length(disease, chrom)

                combi_scaled_rm[offset:offset + chromo_length] = postprocess_weights_without_avg(combi_raw_rm[offset:offset+chromo_length],p_svm)
                combi_avg_rm[offset:offset + chromo_length] = moving_average(
                    combi_scaled_rm[offset:offset + chromo_length],
                    filter_window_size,
                    real_p_pnorm_filter)

                deepcombi_scaled_rm[offset:offset + chromo_length] = postprocess_weights_without_avg(deepcombi_raw_rm[offset:offset+chromo_length],p_svm)
                deepcombi_avg_rm[offset:offset + chromo_length] = moving_average(
                    deepcombi_scaled_rm[offset:offset + chromo_length],
                    filter_window_size,
                    real_p_pnorm_filter)
                offset += chromo_length

            assert not np.isnan(combi_scaled_rm.sum())
            assert not np.isnan(combi_avg_rm.sum())
            assert not np.isnan(deepcombi_scaled_rm.sum())
            assert not np.isnan(deepcombi_avg_rm.sum())

            np.save(os.path.join(FINAL_RESULTS_DIR, 'combi_avg_rm', disease), combi_avg_rm)
            np.save(os.path.join(FINAL_RESULTS_DIR, 'combi_scaled_rm', disease), combi_scaled_rm)
            np.save(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_avg_rm', disease), deepcombi_avg_rm)
            np.save(os.path.join(FINAL_RESULTS_DIR, 'deepcombi_scaled_rm', disease), deepcombi_scaled_rm)

