import os
from parameters_complete import *


# Assertions

assert(noise_snps  % 2 == 0)

#  Data Extraction  ###
workdir = os.environ['WORKDIR']

generate_small_crohn(individuals, inform_snps, noise_snps, rep_inform)
srcfile_inform = workdir+'scripts/marius/tmp/crohn_informative.txt'
srcfile_noise = workdir +'scripts/marius/tmp/crohn_noise.txt'

#  Initialization  ###

num_snps=inform_snps + noise_snps

if use_seed == 1:
	rand('seed', seed)
	randn('seed', seed)


pvalue_no_screening=zeros(rep, num_snps)
pvalue_filter_screening=zeros(rep, num_snps)
pvalue_SVM_screening=zeros(rep, num_snps)
pvalue_SVM_filter_screening=zeros(rep, num_snps)
pvalue_SVM_filter_screening_multisplit=zeros(rep, num_snps)
pvalue_SVM_filter_screening_singlesplit=zeros(rep, num_snps)
roshan_scores=zeros(rep, num_snps)

for i=1: rep
	i
	#  a.	Generiere semi-synthetische Daten gem�� log. Regressionsmodell
	# Data generation based on only one associated SNP

	### Data Generation ###
	# based on just one associated SNPs
	tic
	inform_data=load_data(srcfile_inform)
	offset=3 * round(inform_snps*(rep_inform-1)*rand)
	inform_data = inform_data(: , offset+1: offset+3*inform_snps)

	noise_data=load_data(srcfile_noise)
	data=[noise_data(:, 1:noise_snps*3/2) inform_data noise_data(:, noise_snps*3/2+1:noise_snps*3)]
	labels=generate_toy_data_just_one(inform_data, noise_data, 1, noise_parameter)
	# time.data_generation(i) = toc

	# # based on 20 associated SNPs
	# ### Randomize weights using Dirichlet ###
	# a = [1.1:0.1:1.1+(inform_snps-1)*0.1].^([3.1:0.1:3.1+(inform_snps-1)*0.1])
	# param = dirichletrnd(a)
	# #param(inform_snps+1) = toy_parameter
	# param =  permute_matrix(param,1:length(param),randperm(length(param)),2)

	# ### Data Generation ###
	# tic
	# inform_data = load_data(srcfile_inform)
	# offset = 3 * round(inform_snps*(rep_inform-1)*rand)
	# inform_data = inform_data(:,offset+1:offset+3*inform_snps)

	# noise_data = load_data(srcfile_noise)
	# data = [noise_data(:,1:noise_snps*3/2) inform_data noise_data(:,noise_snps*3/2+1:noise_snps*3)]
	# labels = generate_toy_data(inform_data,noise_data,param, noise_parameter)

	# Multiple Testing with no Screening
	pvalue_no_screening_=chi_square_goodness_of_fit_test(data, labels)

	# Multiple Testing with Filter Screening
	pvalue_filter_screening_=filter_screening(
	    pvalue_no_screening_, top_k, filter_window_size_mtest, p_pnorm_filter_mtest)

	# Multiple Testing with SVM Screening
	# COMBI Method - Multiple Testing with SVM and Filter Screening
	 [pvalue_SVM_screening_, pvalue_SVM_filter_screening_]=combi_method(
	     data, labels, top_k, pnorm_feature_scaling, svm_rep, Cs, p_svm, classy, filter_window_size, p_pnorm_filter)

	# Roshan
	[roshan_scores_]=roshan_method(pvalue_no_screening_, data, labels, top_k_roshan_1, top_k_roshan_2,
	                               pnorm_feature_scaling, svm_rep, Cs, p_svm, classy, filter_window_size, p_pnorm_filter)

	# COMBI Method with Single and Multisplit from B�hlmann & Meinshausen
	pvalue_SVM_filter_screening_multisplit_=multi_split(
	    data, labels, num_splits, top_k, pnorm_feature_scaling, svm_rep, Cs, p_svm, classy, filter_window_size, p_pnorm_filter, gammas_multisplit)
	pvalue_SVM_filter_screening_singlesplit_=single_split(
	    data, labels, top_k, pnorm_feature_scaling, svm_rep, Cs, p_svm, classy, filter_window_size, p_pnorm_filter)

	# Save all pvalues!
  	pvalue_no_screening(i, :) = pvalue_no_screening_
	pvalue_filter_screening(i, :) = pvalue_filter_screening_
	pvalue_SVM_screening(i, :) = pvalue_SVM_screening_
	pvalue_SVM_filter_screening(i, :) = pvalue_SVM_filter_screening_
	roshan_scores(i, :) = roshan_scores_
	pvalue_SVM_filter_screening_multisplit(i, :)= pvalue_SVM_filter_screening_multisplit_
	pvalue_SVM_filter_screening_singlesplit(i, :)=pvalue_SVM_filter_screening_singlesplit_

  clearvars a param inform_data noise_data data res
	save(savefile)
  end
clearvars featmat res result tables


return
