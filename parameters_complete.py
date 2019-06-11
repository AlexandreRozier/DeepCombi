
import numpy as np
import os

if 'ROOT_DIR' not in os.environ:
    os.environ['ROOT_DIR'] = "/home/hx/Work/Masterarbeit/PythonImplementation"

if 'PREFIX' not in os.environ:
    os.environ['PREFIX'] = "default"


ROOT_DIR = os.environ['ROOT_DIR']
DATA_DIR = os.path.join(ROOT_DIR,'data')
TEST_DIR = os.path.join(ROOT_DIR,'tests')
TALOS_OUTPUT_DIR = os.path.join(TEST_DIR,'talos_output')
PARAMETERS_DIR = os.path.join(TEST_DIR,'parameters')


top_k= 30
top_k_multi = top_k*3
top_k_roshan_2= top_k
top_k_roshan_1= top_k_roshan_2*5

alpha_sig = 0.05
rep= 2,
num_splits = 10

# Other params

svm_rep= 1
svm_rep_permtest = 1
Cs= 0.0022  # logspace(-4,4,7), #Cs=logspace(-4,-1,3),
use_seed = 1 # 0 or 1
seed = 13
use_scaling = 1
use_filter = 1

pnorm_feature_scaling = 2 # 0.5, 1, 2, 10, 20, Inf =no scaling
filter_window_size= 35  # 35 # 1-41, odd!!!
p_pnorm_filter = 2 # 1,2,4,100, 0.25, 0.5
classy= 'LIBLINEAR_L2R_L1LOSS_SVC_DUAL'  # Can be: LIBLINEAR_L2R_LR, LIBLINEAR_L2R_L2LOSS_SVC_DUAL, LIBLINEAR_L2R_L2LOSS_SVC, LIBLINEAR_L2R_L1LOSS_SVC_DUAL

filter_window_size_mtest = 9 # 1-41, odd!!!
p_pnorm_filter_mtest = 6# 1,2,4,100, 0.25, 0.5

p_svm = 2
svm_epsilon = 1e-3

rep_inform = 1
inform_snps= 20  # 15-20
noise_snps= 10000  # 10,000, even
n_total_snps = inform_snps + noise_snps
individuals= 300  # 100-300
noise_parameter = 6

roc_ub = 1
prc_ub = 1
# varying t to control for FWER
# start:step:end
t_vector = [
    np.arange(0.000000001, 0.000001, 0.000000001),
    np.arange(0.000001, 0.001, 0.000001),
    np.arange(0.001, 1, 0.001)]

gammas_multisplit = np.arange(0.05, 1, 0.0001)
