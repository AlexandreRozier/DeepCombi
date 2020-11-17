# qlogin -l cuda=1
# cd DeepCombi
# nvidia-smi
# CUDA_VISIBLE_DEVICES=4 python -m pytest -s tests/test_find_good_settings_CD5.py::TestLOTR_CD5::test_hpsearch_CD5_newscaling

# First job in bash 2 (time: 1d14h, 48 combinations): 			'n_snps': [fm.shape[1]],'epochs': [25, 50, 75, 100], #[25, 50, 75, 100,600]'dropout_rate': [0.3],'l1_reg': [0.1, 0.01, 0.001],#[0.1, 0.01, 0.001, 1.e-04,  1.e-05,1.e-06],'l2_reg': [0],'hidden_neurons': [3, 6, 10, 64],'lr': [0.00001],#[0.00001,0.0001, 0.001 , 0.01],

# Second job in bash 0 (time: 3d22h, 432 combinations ): 			'n_snps': [fm.shape[1]],'epochs': [50, 100,600], #[25, 50, 75, 100,600]'dropout_rate': [0.3, 0.5],'l1_reg': [0.1, 0.01, 0.001],#[0.1, 0.01, 0.001, 1.e-04,  1.e-05,1.e-06],'l2_reg': [0.1, 0.01, 0.001],'hidden_neurons': [10, 64],'lr': [0.000001, 0.00001,0.001 , 0.01],#[0.00001,0.0001, 0.001 , 0.01],

# Job with new scaling (64 combinations, time: 3:52:23): 	params_space = {
#			'n_snps': [fm.shape[1]],
#			'epochs': [50, 100], #[25, 50, 75, 100,600]
#			'dropout_rate': [0.3],
#			'l1_reg': [ 0.01, 0.001, 0.0001, 0],#[0.1, 0.01, 0.001, 1.e-04,  1.e-05,1.e-06],
#			'l2_reg': [0.01, 0.001, 0.0001,0],
#			'hidden_neurons': [ 64],
#			'lr': [0.00001, 0.000001],#[0.00001,0.0001, 0.001 , 0.01],
#		}

# CUDA_VISIBLE_DEVICES=0 python -m pytest -s tests/test_find_good_settings_CD5.py::TestLOTR_CD5::test_hpsearch_CD3_weights



import os
import numpy as np
from time import time
import matplotlib
import pdb
matplotlib.use('Agg')

import pandas as pd
import pickle
import tensorflow
from models import create_montaez_dense_model, best_params_montaez
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from helpers import char_matrix_to_featmat, get_available_gpus
from parameters_complete import disease_IDs

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight


from keras import backend as K
from combi import permuted_deepcombi_method, real_classifier
from tqdm import tqdm
import talos
from talos.utils.gpu_utils import parallel_gpu_jobs

#from talos.metrics.keras_metrics import fbeta_score

TEST_PERCENTAGE = 0.20
seed = 6666 # Satan helps us
random_state = np.random.RandomState(seed)

if 'ROOT_DIR' not in os.environ:
    os.environ['ROOT_DIR'] = "/home/bmieth/DeepCombi"

if 'PREFIX' not in os.environ:
    os.environ['PREFIX'] = "default"


disease_IDs = ['CD', 'BD','CAD','HT','RA','T1D','T2D']
diseases = ['Crohns disease','Bipolar disorder', 'Coronary artery disease','Hypertension','Rheumatoid arthritis','Type 1 Diabetes','Type 2 diabetes']
     

ROOT_DIR = os.environ['ROOT_DIR']
SYN_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'synthetic')
REAL_DATA_DIR = os.path.join(ROOT_DIR,'data','WTCCC')
TEST_DIR = os.path.join(ROOT_DIR,'tests')
IMG_DIR = os.path.join(ROOT_DIR,'img')
TALOS_OUTPUT_DIR = os.path.join(TEST_DIR,'talos_output')
PARAMETERS_DIR = os.path.join(TEST_DIR,'parameters')
SAVED_MODELS_DIR = os.path.join(TEST_DIR,'exported_models')
TB_DIR = os.path.join(TEST_DIR,'exported_models')
NUMPY_ARRAYS = os.path.join(ROOT_DIR,'numpy_arrays')
#FINAL_RESULTS_DIR = os.path.join(ROOT_DIR,'experiments','MONTAEZ_preselected_SNPs')
FINAL_RESULTS_DIR = os.path.join(ROOT_DIR,'experiments','MONTAEZ_findCD3_weights_pthresh1e2')

real_pnorm_feature_scaling = 6
pvalue_threshold = 1e-2#1.1, 1e-4
# 1.2 ist dasselbe wie Talos / CD / 5 ... nur mit metrics fbeta
# 1.3 ist dasselbe wie Talos / CD / 5 ... nur mit metrics roc_auc_score

class TestLOTR_CD5(object):


	def test_hpsearch_CD5(self, real_genomic_data, real_labels_cat, real_idx, real_pvalues):
		""" Runs HP search for a subset of chromosomes
		"""
		disease = disease_IDs[int(1-1)]

		chrom = 5

		# 1. Do hyperparam search on each chromosome and find parameters with BEST VAL ACCURAC
		print("Now processing Chromosome:")
		print(chrom)
		raw_data = real_genomic_data(disease, chrom)

		raw_pvalues = real_pvalues(disease, chrom)
		valid_snps = raw_pvalues < pvalue_threshold
		data = raw_data[:,valid_snps,:]

		fm = char_matrix_to_featmat(data, '3d',real_pnorm_feature_scaling)
		labels_cat = real_labels_cat(disease)
		idx = real_idx(disease)
		
		params_space = {
			'n_snps': [fm.shape[1]],
			'epochs': [25, 50, 75, 100], #[25, 50, 75, 100,600]
			'dropout_rate': [0.3],
			'l1_reg': [0.1, 0.01, 0.001],#[0.1, 0.01, 0.001, 1.e-04,  1.e-05,1.e-06],
			'l2_reg': [0],
			'hidden_neurons': [3, 6, 10, 64],
			'lr': [0.00001],#[0.00001,0.0001, 0.001 , 0.01],
		}
		def talos_wrapper(x, y, x_val, y_val, params):
			model = create_montaez_dense_model(params)
			out = model.fit(x=x,
							y=y,
							validation_data=(x_val, y_val),
							epochs=params['epochs'],
							verbose=0)
			return out, model

		nb_gpus = get_available_gpus()

		if nb_gpus == 1:
			parallel_gpu_jobs(0.33)

		os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)

		talos.Scan(x=fm[idx.train],
				y=labels_cat[idx.train],
				x_val=fm[idx.test],
				y_val=labels_cat[idx.test],
				# reduction_method='gamify',
				# reduction_interval=10,
				# reduction_window=10,
				reduction_metric="roc_auc_score", # reduction_metric='val_acc',
				# reduction_threshold=0.2,
				# round_limit = 100,
				minimize_loss=False,
				params=params_space,
				model=talos_wrapper,
				experiment_name=os.path.join('experiments','MONTAEZ_findCD5','pcut_'+str(pvalue_threshold)))


	def test_hpsearch_CD5_newscaling(self, real_genomic_data, real_labels_cat, real_idx, real_pvalues):
		""" Runs HP search for a subset of chromosomes
		"""
		disease = disease_IDs[int(1-1)]

		chrom = 5

		# 1. Do hyperparam search on each chromosome and find parameters with BEST VAL ACCURAC
		print("Now processing Chromosome:")
		print(chrom)
		raw_data = real_genomic_data(disease, chrom)

		raw_pvalues = real_pvalues(disease, chrom)
		valid_snps = raw_pvalues < pvalue_threshold
		data = raw_data[:,valid_snps,:]

		fm = char_matrix_to_featmat(data, '3d',real_pnorm_feature_scaling)
		# Do centering and scaling like Marina
		fm[fm>0]=1
		fm[fm<0]=0.
		
		mean = np.mean(np.mean(fm,0),0)
		std = np.std(np.std(fm,0),0)
		fm = (fm -mean)/std
		
		labels_cat = real_labels_cat(disease)
		idx = real_idx(disease)
		
		params_space = {
			'n_snps': [fm.shape[1]],
			'epochs': [10,50, 100], #[25, 50, 75, 100,600]
			'dropout_rate': [0.3],
			'l1_reg': [ 0.01, 0.001, 0.0001, 0],#[0.1, 0.01, 0.001, 1.e-04,  1.e-05,1.e-06],
			'l2_reg': [0.01, 0.001, 0.0001,0],
			'hidden_neurons': [ 64],
			'lr': [0.00001, 0.000001],#[0.00001,0.0001, 0.001 , 0.01],
		}

		def talos_wrapper(x, y, x_val, y_val, params):
			model = create_montaez_dense_model(params)
			out = model.fit(x=x,
							y=y,
							validation_data=(x_val, y_val),
							epochs=params['epochs'],
							verbose=0)
			return out, model

		nb_gpus = get_available_gpus()

		if nb_gpus == 1:
			parallel_gpu_jobs(0.33)

		os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)

		talos.Scan(x=fm[idx.train],
				y=labels_cat[idx.train],
				x_val=fm[idx.test],
				y_val=labels_cat[idx.test],
				# reduction_method='gamify',
				# reduction_interval=10,
				# reduction_window=10,
				#reduction_metric="roc_auc_score", # reduction_metric='val_acc',
				# reduction_threshold=0.2,
				# round_limit = 100,
				#minimize_loss=False,
				params=params_space,
				model=talos_wrapper,
				experiment_name=os.path.join('experiments','MONTAEZ_findCD5','newscaling'))
				

	def test_hpsearch_CD3_weights(self, real_genomic_data, real_labels_cat, real_idx, real_pvalues):
		""" Runs HP search for a subset of chromosomes
		"""
		disease = disease_IDs[int(1-1)]

		chrom = 3

		# 1. Do hyperparam search on each chromosome and find parameters with BEST VAL ACCURAC
		print("Now processing Chromosome:")
		print(chrom)
		raw_data = real_genomic_data(disease, chrom)

		raw_pvalues = real_pvalues(disease, chrom)
		valid_snps = raw_pvalues < pvalue_threshold
		data = raw_data[:,valid_snps,:]

		fm = char_matrix_to_featmat(data, '3d',real_pnorm_feature_scaling)
		# Do centering and scaling like Marina
		fm[fm>0]=1
		fm[fm<0]=0.
		
		mean = np.mean(np.mean(fm,0),0)
		std = np.std(np.std(fm,0),0)
		fm = (fm -mean)/std
		
		labels_cat = real_labels_cat(disease)
		idx = real_idx(disease)
		
		params_space = {
			'n_snps': [fm.shape[1]],
			'epochs': [100,500,1000],
			'dropout_rate': [0.3],
			'l1_reg': [0.1,0.01,0.001, 0.0001],
			'l2_reg': [0.01, 0.001,0.0001, 0],
			'hidden_neurons': [64],
			'lr': [0.0001, 0.00001, 0.000001, 0.0000001],
		}

		def talos_wrapper(x, y, x_val, y_val, params):
			model = create_montaez_dense_model(params)
			y_integers = np.argmax(y, axis=1)
			class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
			d_class_weights = dict(enumerate(class_weights))
			out = model.fit(x=x,
							y=y,
							validation_data=(x_val, y_val),
							epochs=params['epochs'],
							verbose=0,
							class_weight=d_class_weights)
			return out, model

		nb_gpus = get_available_gpus()

		if nb_gpus == 1:
			parallel_gpu_jobs(0.33)

		os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)

		talos.Scan(x=fm[idx.train],
				y=labels_cat[idx.train],
				x_val=fm[idx.test],
				y_val=labels_cat[idx.test],
				# reduction_method='gamify',
				# reduction_interval=10,
				# reduction_window=10,
				#reduction_metric="roc_auc_score", # reduction_metric='val_acc',
				# reduction_threshold=0.2,
				# round_limit = 100,
				#minimize_loss=False,
				params=params_space,
				model=talos_wrapper,
				experiment_name=os.path.join('experiments','MONTAEZ_findCD3_weights_pthresh1e2'))

				

	def test_hpsearch_CD5_5050(self, real_genomic_data, real_labels_cat, real_idx, real_pvalues):
		""" Runs HP search for a subset of chromosomes
		"""
		disease = disease_IDs[int(1-1)]

		chrom = 5

		# 1. Do hyperparam search on each chromosome and find parameters with BEST VAL ACCURAC
		print("Now processing Chromosome:")
		print(chrom)
		raw_data = real_genomic_data(disease, chrom)

		raw_pvalues = real_pvalues(disease, chrom)
		valid_snps = raw_pvalues < pvalue_threshold
		data = raw_data[:,valid_snps,:]
		
		fm_now = char_matrix_to_featmat(data, '3d',real_pnorm_feature_scaling)
		labels_cat_now = real_labels_cat(disease)
		num_neg_now =  sum(labels_cat_now[:,1])

		ind_pos = np.where(labels_cat_now[:,0])
		ind_neg = np.where(labels_cat_now[:,1])

		fm = fm_now[np.concatenate([ind_neg[0], ind_pos[0][:num_neg_now.astype(int)]]),:,:]
		n_subjects = int(fm.shape[0])
		n_snps = int(fm.shape[1])

		labels_cat = labels_cat_now[np.concatenate([ind_neg[0], ind_pos[0][:num_neg_now.astype(int)]]),:]


		# Create train and test sets

		splitter =  StratifiedShuffleSplit(n_splits=1, test_size = TEST_PERCENTAGE, random_state=random_state)
		idx_train, idx_test = next(splitter.split(np.zeros(n_subjects), labels_cat))

		params_space = {
			'n_snps': [fm.shape[1]],
			'epochs': [25, 50, 75, 100], #[25, 50, 75, 100,600]
			'dropout_rate': [0.3],
			'l1_reg': [0.1, 0.01, 0.001],#[0.1, 0.01, 0.001, 1.e-04,  1.e-05,1.e-06],
			'l2_reg': [0],
			'hidden_neurons': [3, 6, 10, 64],
			'lr': [0.00001],#[0.00001,0.0001, 0.001 , 0.01],
		}
		def talos_wrapper(x, y, x_val, y_val, params):
			model = create_montaez_dense_model(params)
			out = model.fit(x=x,
							y=y,
							validation_data=(x_val, y_val),
							epochs=params['epochs'],
							verbose=0)
			return out, model

		#nb_gpus = get_available_gpus()

		#if nb_gpus == 1:
		#	parallel_gpu_jobs(0.33)

		os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)

		talos.Scan(x=fm[idx_train],
				y=labels_cat[idx_train],
				x_val=fm[idx_test],
				y_val=labels_cat[idx_test],
				# reduction_method='gamify',
				# reduction_interval=10,
				# reduction_window=10,
				reduction_metric="roc_auc_score", # reduction_metric='val_acc',
				# reduction_threshold=0.2,
				# round_limit = 100,
				minimize_loss=False,
				params=params_space,
				model=talos_wrapper,
				experiment_name=os.path.join('experiments','MONTAEZ_findCD5','5050_1'))

	def test_extract_best_hps_CD5(self):
		disease_id = disease_IDs[int(1-1)]

		try:
			data = pd.DataFrame()
			talos_disease_directory = os.path.join(FINAL_RESULTS_DIR, 'talos', disease_id)
			for root, _, files in os.walk(talos_disease_directory):
				for file in files:
					if file.endswith(".csv"):
						df = pd.read_csv(os.path.join(root, file))
						data = data.append(df, ignore_index=True)

			data.sort_values(by=['val_acc'], ascending=False, inplace=True)
			best_hps = data[data['acc'] > 0.80].iloc[0].to_dict()
			#best_hps['epochs'] = 250
			#best_hps['hidden_neurons'] = 6
			#best_hps['lr'] = 1e-4
			#best_hps['l1_reg'] = 1e-5
			print(best_hps)
			chromo = 5

			filename = os.path.join(FINAL_RESULTS_DIR, 'hyperparams', disease_id, 'chrom{}.p'.format(chromo))
			os.makedirs(os.path.dirname(filename), exist_ok=True)
			pickle.dump(best_hps, open(filename, 'wb'))
		except Exception as identifier:
			print('Failed for item {}. Reason:{}'.format(disease_id, identifier))
			raise ValueError(identifier)
			
	def test_train_models_with_best_params_CD5(self, real_genomic_data, real_labels_cat, real_idx):
		""" Generate a per-chromosom trained model for futur LRP-mapping quality assessment
		TRAINS ON WHOLE DATASET
		"""
		disease_id = disease_IDs[int(1-1)]
		chrom = 5



		# Load data, hp & labels
		data = real_genomic_data(disease_id, chrom)
		fm = char_matrix_to_featmat(data, '3d', real_pnorm_feature_scaling)

		labels_cat = real_labels_cat(disease_id)

		hp = pickle.load(open(os.path.join(FINAL_RESULTS_DIR, 'hyperparams', disease_id, 'chrom{}.p'.format(chrom)), 'rb'))
		print(hp)

		hp['epochs'] = int(hp['epochs'])
		hp['n_snps'] = int(fm.shape[1])
		#hp['epochs'] = 250
		#hp['hidden_neurons'] = 6
		#hp['lr'] = 1e-4
		#hp['l1_reg'] = 1e-5 # TODO change me back
		# Train
		os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'csv_logs', disease_id), exist_ok=True)

		model = create_montaez_dense_model(hp)
		model.fit(x=fm,
					y=labels_cat,
					epochs=hp['epochs'],
					callbacks=[
						CSVLogger(os.path.join(FINAL_RESULTS_DIR, 'csv_logs', disease_id, '{}'.format(chrom)))
					],
					verbose=0)
					
		# Calculate AUC from the best model
		y_pred = model.predict(x=fm)

		#print(classification_report(np.argmax(Y, axis=-1), np.argmax(y_pred, axis=-1), output_dict=False))

		auc = roc_auc_score(labels_cat, y_pred, average='weighted')
		prec = average_precision_score(labels_cat, y_pred, average='weighted')
		print('ROC AUC score for best model: %f' % auc)	
		print('Prec-Rec AUC score for best model: %f' % prec)	
		
		filename = os.path.join(FINAL_RESULTS_DIR, 'trained_models', disease_id, 'model{}.h5'.format(chrom))
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		model.save(filename)
		K.clear_session()
		del data, fm, model


	def test_train_models_with_constant_params_CD5(self, real_genomic_data, real_labels_cat, real_idx):

		disease_id = disease_IDs[int(1-1)]
		chrom = 5

		# Load data, hp & labels
		data = real_genomic_data(disease_id, chrom)
		fm = char_matrix_to_featmat(data, '3d', real_pnorm_feature_scaling)

		labels_cat = real_labels_cat(disease_id)
		
		idx = real_idx(disease_id)


		hp =  {'dropout_rate': 0.3, 'epochs': 75, 'hidden_neurons': 64, 'l1_reg': 0.001, 'l2_reg': 0.0, 'lr': 1e-05, 'n_snps': int(fm.shape[1])}

		print(hp)
				
		# Train
		#os.makedirs(os.path.join(FINAL_RESULTS_DIR, 'csv_logs', disease_id), exist_ok=True)

		model = create_montaez_dense_model(hp)
		model.fit(x=fm[idx.train],
					y=labels_cat[idx.train],
					epochs=hp['epochs'],
					#callbacks=[
					#	CSVLogger(os.path.join(FINAL_RESULTS_DIR, 'csv_logs', disease_id, '{}'.format(chrom)))
					#],
					verbose=1)
					
		# Calculate AUC from the model
		y_pred = model.predict(x=fm[idx.test])
		pdb.set_trace()
		auc = roc_auc_score(labels_cat[idx.test], y_pred, average='weighted')
		prec = average_precision_score(labels_cat, y_pred, average='weighted')
		print('ROC AUC score for best model: %f' % auc)	
		print('Prec-Rec AUC score for best model: %f' % prec)	
		
		K.clear_session()
		del data, fm, model
