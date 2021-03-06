
import os
import numpy as np
from time import time

if 'ROOT_DIR' not in os.environ:
    os.environ['ROOT_DIR'] = "/home/bmieth/DeepCombi"

if 'PREFIX' not in os.environ:
    os.environ['PREFIX'] = "default"


disease_IDs = ['BD','CAD','CD', 'HT','RA','T1D','T2D']
diseases = ['Bipolar disorder', 'Coronary artery disease','Crohns disease','Hypertension','Rheumatoid arthritis','Type 1 Diabetes','Type 2 diabetes']
     

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
FINAL_RESULTS_DIR = os.path.join(ROOT_DIR,'experiments','MONTAEZ_final')


pvalue_threshold = 1e-2#1.1, 1e-4


########
ttbr = 6
syn_n_subjects= 300
inform_snps= 20  # 15-20
noise_snps= 10000  # 10,000, even
n_total_snps = inform_snps + noise_snps
top_k= 30
real_top_k = 100 # This one

################################
# DeepCOMBI postprocessing parameters
filter_window_size_deep=  21 # 21
real_p_pnorm_filter_deep = 2	# 2
p_svm_deep = 2	# 2
real_top_k_deep = 200	# 200

# best for LRPAlpha2Beta1: 21, 2,2,200
# best for LRPAlpha2Beta1IgnoreBias: 35, 2,1,200
# best for LRPZPlusFast: 35, 1 ,2 , 400
################################

n_permutations = 100
################################
		
		
nb_of_jobs = 31


####
top_k_multi = top_k*3
top_k_roshan_2= top_k
top_k_roshan_1= top_k_roshan_2*5

alpha_sig_toy = 0.05
num_splits = 10

# Other params

svm_rep= 1
svm_rep_permtest = 1
Cs= 0.0022  
real_Cs= 1e-5

seed = 6666 # Satan helps us
random_state = np.random.RandomState(seed)


filter_window_size= 35  # THIS ONE
 # 35 # 1-41, odd!!!
#filter_window_size_mtest = 9  # 35 # 1-41, odd!!!

# USED in char_matrix_to_featmat to scale the resulting feature matrix
pnorm_feature_scaling = 2 
real_pnorm_feature_scaling = 6

# USED TO APPLY SCALING ON WEIGHTS DURING THE POSTPROCESSING STEP 
p_pnorm_filter = 2 
real_p_pnorm_filter = 2 # THIS ONE
#real_p_pnorm_filter_mtest =6


p_svm = 2 # THIS ONE # Used on SVM and LRP weights before applying postprocessing function 
svm_epsilon = 1e-3

rep_inform = 1

# varying t to control for FWER
# start:step:end
thresholds = np.hstack((np.arange(10e-9,10e-6,10e-9), np.arange(10e-6,10e-3,10e-6), np.arange(10e-3,10e-0,10e-3)))

gammas_multisplit = np.arange(0.05, 1, 0.0001)
