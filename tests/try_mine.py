# qlogin
# cd DeepCombi/tests


import os
import numpy as np
from time import time

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pdb

import pandas as pd
import pickle
import tensorflow

from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, accuracy_score
from sklearn.utils import class_weight
from sklearn.svm import LinearSVC

from keras import backend as K
import keras
import keras.callbacks
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Conv1D, Flatten, Activation, AveragePooling1D, MaxPooling1D
from keras.models import Sequential, load_model
from keras.regularizers import l1_l2
from sklearn.model_selection import StratifiedShuffleSplit

import sys
sys.path.insert(1, '/home/bmieth/DeepCombi/')

from helpers import chi_square, postprocess_weights, char_matrix_to_featmat
import innvestigate
import innvestigate.utils as iutils
import scipy
import h5py


from conftest import real_genomic_data

if 'ROOT_DIR' not in os.environ:
    os.environ['ROOT_DIR'] = "/home/bmieth/DeepCombi"

if 'PREFIX' not in os.environ:
    os.environ['PREFIX'] = "default"

TEST_PERCENTAGE = 0.10
seed = 6666 # Satan helps us
pvalue_threshold = 1e-2#1.1, 1e-4

random_state = np.random.RandomState(seed)
# pvalue_threshold = 1.3

# Load data
data_orig = scipy.io.loadmat('/home/bmieth/DeepCombi/data/WTCCC/CD/chromo_3_processed.mat')['X']

try:
	labels_orig = scipy.io.loadmat('/home/bmieth/DeepCombi/data/WTCCC/CD/labels.mat')['y'][0]
except Exception:
	labels_orig = h5py.File('/home/bmieth/DeepCombi/data/WTCCC/CD/labels.mat', 'r').get('y')[:].T[0]
	
# pvalues
rpvt_pvalues = chi_square(data_orig, labels_orig)

# pvalue thresholding
valid_snps = rpvt_pvalues < pvalue_threshold
data = data_orig[:,valid_snps,:]

fm_orig = char_matrix_to_featmat(data, '3d', 6)	

labels_cat = tensorflow.keras.utils.to_categorical((labels_orig+1)/2)

# To disable class balance in data
#num_neg_now =  sum(labels_cat[:,1])
#ind_pos = np.where(labels_cat[:,0])
#ind_neg = np.where(labels_cat[:,1])
#fm_orig = fm_orig[np.concatenate([ind_neg[0], ind_pos[0][:num_neg_now.astype(int)]]),:,:]
#labels_cat = labels_cat[np.concatenate([ind_neg[0], ind_pos[0][:num_neg_now.astype(int)]]),:]


n_subjects = int(fm_orig.shape[0])
n_snps = int(fm_orig.shape[1])

hp =  {'dropout_rate': 0.3, 'epochs': 500, 'hidden_neurons': 64, 'l1_reg': 0.001, 'l2_reg': 0.0001, 'lr': 1e-05, 'n_snps': n_snps} 

# opt params with pthresh 1e-02: hp =  {'dropout_rate': 0.3, 'epochs': 500, 'hidden_neurons': 64, 'l1_reg': 0.001, 'l2_reg': 0.0001, 'lr': 1e-05, 'n_snps': n_snps} 

# opt params without pthresh: hp =  {'dropout_rate': 0.3, 'epochs': 100, 'hidden_neurons': 64, 'l1_reg': 0.01, 'l2_reg': 0.0001, 'lr': 1e-06, 'n_snps': n_snps}  

# Do centering and scaling like Marina
fm = fm_orig
fm[fm>0]=1
fm[fm<0]=0.

mean = np.mean(np.mean(fm,0),0)
std = np.std(np.std(fm,0),0)

fm = (fm-mean)/std

# Create train and test sets

splitter =  StratifiedShuffleSplit(n_splits=1, test_size = TEST_PERCENTAGE, random_state=random_state)
idx_train, idx_test = next(splitter.split(np.zeros(n_subjects), (labels_orig+1)/2))

X_train = fm[idx_train]
X_test = fm[idx_test]

X_train_orig = fm_orig[idx_train]
X_test_orig = fm_orig[idx_test]

# Create model

model=Sequential()
model.add(Flatten(input_shape=(int(hp['n_snps']), 3)))

model.add(Dense(activation='relu',
				units=int(hp['hidden_neurons']),
				kernel_regularizer=l1_l2(
					l1=hp['l1_reg'], l2=hp['l2_reg']
				))
		)

model.add(Dropout(hp['dropout_rate']))

model.add(Dense(activation='relu',
				units=int(hp['hidden_neurons']),
				kernel_regularizer=l1_l2(
					l1=hp['l1_reg'], l2=hp['l2_reg']
				))
		)
		
model.add(Dropout(hp['dropout_rate']))
model.add(Dense(activation='softmax',
				units=2,
				kernel_regularizer=l1_l2(
					l1=hp['l1_reg'], l2=hp['l2_reg']
				))
		)

# Compile
model.compile(loss='categorical_crossentropy',
				optimizer=optimizers.Adam(lr=hp['lr']), 
				metrics=['categorical_accuracy'])

# Create class weights
y_integers = np.argmax(labels_cat[idx_train], axis=1)
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))

# Train
history = model.fit(x=X_train, y=labels_cat[idx_train], validation_data=(X_test, labels_cat[idx_test]), epochs=hp['epochs'], verbose=1, class_weight=d_class_weights)
#, class_weight=d_class_weights
			
model.summary()
# Evaluate
y_pred = model.predict(x=X_test)
auc = roc_auc_score(labels_cat[idx_test], y_pred, average='weighted')
prec = average_precision_score(labels_cat[idx_test], y_pred, average='weighted')

acc = accuracy_score(labels_cat[idx_test][:,0], y_pred[:,0]>y_pred[:,1])
bal_acc = balanced_accuracy_score(labels_cat[idx_test][:,0], y_pred[:,0]>y_pred[:,1])
num_pos =  sum(labels_cat[idx_test][:,0])
num_neg =  sum(labels_cat[idx_test][:,1])
num_pos_pred = sum(y_pred[:,0]>y_pred[:,1])
num_neg_pred = sum(y_pred[:,0]<y_pred[:,1])

print(hp)
print('Number of true positives: %i' % num_pos)	
print('Number of true negatives: %i' % num_neg)	
print('Number of positive predicted: %i' % num_pos_pred)	
print('Number of negative predicted: %i' % num_neg_pred)	
print('Accuracy: %f' % acc)
print('Balanced Accuracy: %f' % bal_acc)
print('ROC AUC score: %f' % auc)	
print('Prec-Rec AUC score: %f' % prec)	
print('class weights:')
print(d_class_weights)


# Train SVM for comparison on original data (no pthresh)
fm_svm = char_matrix_to_featmat(data_orig, '3d', 6)	
#fm_svm[fm_svm>0]=1
#fm_svm[fm_svm<0]=0.

#mean = np.mean(np.mean(fm_svm,0),0)
#std = np.std(np.std(fm_svm,0),0)

#fm_svm = (fm_svm-mean)/std
X_train_svm = fm_svm[idx_train]
X_test_svm = fm_svm[idx_test]

#clf = LinearSVC(penalty='l1', C=.01, dual=False, tol=1e-1,fit_intercept=True, class_weight=None, verbose=0, random_state=0, max_iter=100)
#clf.fit(X=X_train_svm.reshape(len(X_train_svm),-1), y=labels_cat[idx_train][:,0])
#y_pred_svm = clf.predict(X=X_test_svm.reshape(len(X_test_svm),-1))

# best model on alex scaled data
clf = LinearSVC(penalty='l2', loss='hinge', C=1.0000e-05, dual=True, tol=1e-3, verbose=0)
clf.fit(X=X_train_svm.reshape(len(X_train_svm),-1), y=labels_cat[idx_train][:,0])
y_pred_svm = clf.predict(X=X_test_svm.reshape(len(X_test_svm),-1))

# Evaluate
auc = roc_auc_score(labels_cat[idx_test][:,0], y_pred_svm, average='weighted')
prec = average_precision_score(labels_cat[idx_test][:,0], y_pred_svm, average='weighted')
acc = accuracy_score(labels_cat[idx_test][:,0], y_pred_svm)
bal_acc = balanced_accuracy_score(labels_cat[idx_test][:,0], y_pred_svm)
num_pos =  sum(labels_cat[idx_test][:,0])
num_neg =  len(y_pred_svm)-num_pos
num_pos_pred = sum(y_pred_svm)
num_neg_pred = len(y_pred_svm)-num_pos_pred

print('SVM evaluation:')
print('Number of positive predicted: %i' % num_pos_pred)	
print('Number of negative predicted: %i' % num_neg_pred)	
print('Accuracy: %f' % acc)
print('Balanced Accuracy: %f' % bal_acc)
print('ROC AUC score: %f' % auc)	
print('Prec-Rec AUC score: %f' % prec)	

fig = plt.figure()
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
fig.savefig('/home/bmieth/DeepCombi/tests/accs_and_losses/chr3/reproducewin_acc.png')

fig_2 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
fig_2.savefig('/home/bmieth/DeepCombi/tests/accs_and_losses/chr3/reproducewin_loss.png')

# Explaination - COMBI and DeepCOMBI
#data = scipy.io.loadmat('/home/bmieth/DeepCombi/data/WTCCC/CD/chromo_5_processed.mat')['X']
#labels = scipy.io.loadmat('/home/bmieth/DeepCombi/data/WTCCC/CD/labels.mat')['y'][0]

# combi
svm_weights = clf.coef_[0] # n_snps * 3

top_k = 100
filter_window_size = 35
psvm = 2
pnorm_filter = 2

filter_window_size_deep=  21 # 21
pnorm_filter_deep = 2	# 2
psvm_deep = 2	# 2
top_k_deep = 200	# 200

selected_indices_sorted, post_svm_weights = postprocess_weights(svm_weights,top_k, filter_window_size, psvm, pnorm_filter)
combi_pvalues=np.ones(rpvt_pvalues.shape)

combi_pvalues_not_one = chi_square(data_orig[:,selected_indices_sorted], labels_orig)
combi_pvalues[selected_indices_sorted]=combi_pvalues_not_one

# deepcombi
for i, layer in enumerate(model.layers):
		if layer.name == 'dense_1':
			layer.name = 'blu{}'.format(str(i))
		if layer.name == 'dense_2':
			layer.name = 'bla{}'.format(str(i))
		if layer.name == 'dense_3':
			layer.name = 'bleurg{}'.format(str(i))
			
model = iutils.keras.graph.model_wo_softmax(model)
analyzer = innvestigate.analyzer.LRPAlpha2Beta1(model)

# LRP with newly scaled data
lrp_weights_raw = np.absolute(analyzer.analyze(fm)).sum(0) # n_snps * 3
lrp_weights_raw= lrp_weights_raw.reshape((-1))

# adjust for p-value thresholding
valid_inds = [i for i, x in enumerate(valid_snps) if x]
valid_inds_all = np.sort(np.concatenate(([3*k + 0 for k in valid_inds],[3*k + 1 for k in valid_inds],[3*k + 2 for k in valid_inds])))

lrp_weights=np.zeros((len(rpvt_pvalues)*3,))
lrp_weights[valid_inds_all]=lrp_weights_raw

selected_indices_sorted,post_lrp_weights = postprocess_weights(lrp_weights, top_k_deep, filter_window_size_deep, psvm_deep, pnorm_filter_deep)

deepcombi_pvalues=np.ones(rpvt_pvalues.shape)
deepcombi_pvalues_not_one = chi_square(data_orig[:,selected_indices_sorted], labels_orig)
deepcombi_pvalues[selected_indices_sorted]=deepcombi_pvalues_not_one
 
# LRP with original scaling
analyzer = innvestigate.analyzer.LRPAlpha2Beta1(model)
lrp_weights_orig_raw = np.absolute(analyzer.analyze(fm_orig)).sum(0) # n_snps * 3
lrp_weights_orig_raw= lrp_weights_orig_raw.reshape((-1))

# adjust for p-value thresholding
lrp_weights_orig=np.zeros((len(rpvt_pvalues)*3,))
lrp_weights_orig[valid_inds_all]=lrp_weights_orig_raw

selected_indices_sorted_orig,post_lrp_weights_orig = postprocess_weights(lrp_weights_orig, top_k_deep, filter_window_size_deep, psvm_deep, pnorm_filter_deep)

deepcombi_pvalues_orig=np.ones(rpvt_pvalues.shape)
deepcombi_pvalues_orig_not_one = chi_square(data_orig[:,selected_indices_sorted_orig], labels_orig)
deepcombi_pvalues_orig[selected_indices_sorted_orig]=deepcombi_pvalues_orig_not_one

# LRP on group 1 only
analyzer = innvestigate.analyzer.LRPAlpha2Beta1(model)
lrp_weights_bool1_raw = np.absolute(analyzer.analyze(fm[labels_cat[:,0].astype(bool),:,:])).sum(0) # n_snps * 3
lrp_weights_bool1_raw= lrp_weights_bool1_raw.reshape((-1))

# adjust for p-value thresholding
lrp_weights_bool1=np.zeros((len(rpvt_pvalues)*3,))
lrp_weights_bool1[valid_inds_all]=lrp_weights_bool1_raw

selected_indices_sorted_bool1,post_lrp_weights_bool1 = postprocess_weights(lrp_weights_bool1, top_k_deep, filter_window_size_deep, psvm_deep, pnorm_filter_deep)

deepcombi_pvalues_bool1=np.ones(rpvt_pvalues.shape)
deepcombi_pvalues_not_one_bool1 = chi_square(data_orig[:,selected_indices_sorted_bool1], labels_orig)
deepcombi_pvalues_bool1[selected_indices_sorted_bool1]=deepcombi_pvalues_not_one_bool1

# LRP on group 2 only
analyzer = innvestigate.analyzer.LRPAlpha2Beta1(model)
lrp_weights_bool2_raw = np.absolute(analyzer.analyze(fm[labels_cat[:,1].astype(bool),:,:])).sum(0) # n_snps * 3
lrp_weights_bool2_raw= lrp_weights_bool2_raw.reshape((-1))

# adjust for p-value thresholding
lrp_weights_bool2=np.zeros((len(rpvt_pvalues)*3,))
lrp_weights_bool2[valid_inds_all]=lrp_weights_bool2_raw

selected_indices_sorted_bool2,post_lrp_weights_bool2 = postprocess_weights(lrp_weights_bool2, top_k_deep, filter_window_size_deep, psvm_deep, pnorm_filter_deep)

deepcombi_pvalues_bool2=np.ones(rpvt_pvalues.shape)
deepcombi_pvalues_not_one_bool2 = chi_square(data_orig[:,selected_indices_sorted_bool2], labels_orig)
deepcombi_pvalues_bool2[selected_indices_sorted_bool2]=deepcombi_pvalues_not_one_bool2

#  plot rpvt pvalues, svm weights, combi pvalues, lrp scores, deep combi pvalues
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10)= plt.subplots(10,1, figsize=(20,25))
ax1.plot(-np.log10(rpvt_pvalues), '.')
ax1.set_ylabel('rpvt p')

ax2.plot(np.absolute(svm_weights), '.')
ax2.set_ylabel('svm_w')

ax3.plot(post_svm_weights, '.')
ax3.set_ylabel('post svm_w')

ax4.plot(-np.log10(combi_pvalues), '.')
ax4.set_ylabel('combi p') 

ax5.plot(lrp_weights.reshape(-1,1), '.')
ax5.set_ylabel('lrp_w')

ax6.plot(post_lrp_weights, '.')
ax6.set_ylabel('post lrp_w')

ax7.plot(-np.log10(deepcombi_pvalues), '.')
ax7.set_ylabel('deepcombi p')

ax8.plot(lrp_weights_orig.reshape(-1,1), '.')
ax8.set_ylabel('lrp_w orig')

ax9.plot(post_lrp_weights_orig, '.')
ax9.set_ylabel('post lrp_w orig')

ax10.plot(-np.log10(deepcombi_pvalues_orig), '.')
ax10.set_ylabel('deepcombi p orig')
ax10.set_xlabel('SNPs')

fig.savefig('/home/bmieth/DeepCombi/tests/accs_and_losses/chr3/weights_reproducewin.png')

#  plot rpvt pvalues, svm weights, combi pvalues, lrp scores, deep combi pvalues
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10)= plt.subplots(10,1, figsize=(20,25))
ax1.plot(-np.log10(rpvt_pvalues), '.')
ax1.set_ylabel('rpvt p')

ax2.plot(np.absolute(svm_weights), '.')
ax2.set_ylabel('svm_w')

ax3.plot(post_svm_weights, '.')
ax3.set_ylabel('post svm_w')

ax4.plot(-np.log10(combi_pvalues), '.')
ax4.set_ylabel('combi p') 

ax5.plot(lrp_weights_bool1.reshape(-1,1), '.')
ax5.set_ylabel('lrp_w_bool1')

ax6.plot(post_lrp_weights_bool1, '.')
ax6.set_ylabel('post lrp_w_bool1')

ax7.plot(-np.log10(deepcombi_pvalues_bool1), '.')
ax7.set_ylabel('deepcombi p bool1')

ax8.plot(lrp_weights_bool2.reshape(-1,1), '.')
ax8.set_ylabel('lrp_w bool2')

ax9.plot(post_lrp_weights_bool2, '.')
ax9.set_ylabel('post lrp_w bool2')

ax10.plot(-np.log10(deepcombi_pvalues_bool2), '.')
ax10.set_ylabel('deepcombi p bool2')
ax10.set_xlabel('SNPs')

fig.savefig('/home/bmieth/DeepCombi/tests/accs_and_losses/chr3/weights_bool12_reproducewin.png')

pdb.set_trace()
K.clear_session()
