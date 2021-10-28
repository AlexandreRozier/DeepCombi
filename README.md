# DeepCOMBI: Explainable artificial intelligence for the analysis and discovery in genome-wide association studies

A Python framework for the analysis of GWAS data with special focus on explainable artificial intelligence.

This repository contains an implementation of the DeepCOMBI method from [here](https://www.biorxiv.org/content/10.1101/2020.11.06.371542v1) .
DeepCOMBI is a neural-network-based method to identify SNP trait associations in GWAS datasets. It is an extension of COMBI, an SVM based GWAS tool, which is described [here](https://www.nature.com/articles/srep36671#methods).

This software package also contains methods for generating artificial GWAS data to analyze with DeepCOMBI. 

Developed by Alexandre Rozier and Bettina Mieth.

## Publication

The Python framework and this website are part of a publication currently under peer-review at Nucleic Acids Research. The pre-print article is available [here](https://www.biorxiv.org/content/10.1101/2020.11.06.371542v1). Links to the published paper will be included here once available.

## Abstract

Deep learning has revolutionized data science in many fields by greatly improving prediction performances in comparison to conventional approaches. Recently, explainable artificial intelligence has emerged as a novel area of research that goes beyond pure prediction improvement by extracting knowledge from deep learning methodologies through the interpretation of their results. We investigate such explanations to explore the genetic architectures of phenotypes in genome-wide association studies. Instead of testing each position in the genome individually, the novel three-step algorithm, called DeepCOMBI, first trains a neural network for the classification of subjects into their respective phenotypes. Second, it explains the classifiers’ decisions by applying layerwise relevance propagation as one example from the pool of explanation techniques. The resulting importance scores are eventually used to determine a subset of most relevant locations for multiple hypothesis testing in the third step. The performance of DeepCOMBI in terms of power and precision is investigated on generated datasets and a 2007 study. Verification of the latter is achieved by validating all findings with independent studies published up until 2020. DeepCOMBI is shown to outperform ordinary raw p-value thresholding and other baseline methods. Two novel disease associations (rs10889923 for hypertension, rs4769283 for type 1 diabetes) were identified.

## How to run DeepCOMBI

## Replicating experiments
In the course of our research (from [Mieth et al.](https://www.biorxiv.org/content/10.1101/2020.11.06.371542v1) ) we have investigated the performance of the proposed method in comparison with the most important baseline methods firstly in a simulation study on generated data and secondly on real data (Wellcome Trust Case Control Consortium (2007) Genome-wide association study of 14,000 cases of seven common diseases and 3,000 shared controls. Nature. 447(7145), 661–678.) To fully reproduce the experiments of our study, please follow the corresponding instructions for the application of DeepCOMBI on both generated and real datasets.

### On generated synthetic datasets
- Run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_data_generation.py::TestDataGeneration::test_synthetic_genotypes_generation --rep 1000` to generate `rep` different genotypes that will be saved in `data/synthetic/genomic.h5py`. Please note, that to generate datasets you need two real datasets to sample from. We use the WTCCC data and randomly select 300 subjects of the Crohn's disease dataset. We draw a random block of 20 consecutive SNPs from chromosome 1 and a random block of 10,000 consecutive SNPs from chromosome 2. The process is described in detail in our manuscript on page 6. Unfortunately, we are not authorized to publish this data and you will have to save your own datasets in the corresponding .mat files. The .mat files should be simple arrays of characters where the number of rows equals the number of subjects and the number of columns equals the number of SNPs * 3 (two letters for the genotype and one space). A small part of it with three subjects and the genotypes of four SNPs given would look like this:
 
AA AA CG GG
AT AA GG GG
TT AT CC GT

Converting your own Plink files should be straightforward.

- Run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_data_generation.py::TestDataGeneration::test_feature_map_generation` to generate the features matrices associated to the genomic datasets previously created in `data/synthetic/genomic.h5py` and saves them in `data/synthetic/2d_fm.h5py` and `data/synthetic/3d_fm.h5py`

- To create Figure 2 of the paper run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_deepcombi.py::TestDeepCOMBI::test_lrp_svm  --rep 1` to plot three exemplary runs. It will be saved in `img_dir/`.

- To create Figure 3 of the paper run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_deepcombi.py::TestDeepCOMBI::test_tpr_fwer_alex  --rep 1000` to plot the performance curves of DeepCOMBI and its competitors.

- To generate Table 1 of the paper run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_deepcombi.py::TestDeepCOMBI::test_svm_cnn_comparison_alex --rep 1000` to investigate the prediction accuracies of the SVM and the DNN on the generated datasets.

### On your own dataset or the 2007 WTCCC dataset
- The data should be saved in the folder `data/`.  The .mat files should be simple arrays of characters where the number of rows equals the number of subjects and the number of columns equals the number of SNPs * 3 (two letters for the genotype and one space). A small part of it with three subjects and the genotypes of four SNPs given would look like this:
 
AA AA CG GG
AT AA GG GG
TT AT CC GT

Converting your Plink files should be straightforward.

- The folder `experiments/:` should contain a subfolder which corresponds to a new NN arcthitecture or hyperparameters setting. 

- Please adjust all file paths and parameters in `parameters_complete.py`.

- Run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_data_generation.py::TestDataGeneration::test_mat_preprocessing` to create (N x d x 2) genomic matrices for all diseases and chromosomes. This only needs to be done once.

- Run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_data_generation.py::TestDataGeneration::test_real_pvalues_generation` to compute and save for each chromosom of each disease the associated p-values (saved in `data/WTCCC/pvalues`). This only needs to be done once.

- Set `FINAL_RESULTS_DIR` in `parameters_complete.py` to indicate where to save the results of this experiment.

- Run `qsub -t 1-154 train_models_with_const_params.sh` to train a DNN on all diseases and chromosomes (7 x 22).

- To save validation accuracies (for Figure 4 of the paper) run `python -m pytest -s tests/test_lotr.py::TestLOTR::test_train_models_save_accuracies`.

- Run `qsub -t 1-7 save_deepcombi_rm.sh` and `qsub -t 1-7 save_combi_rm.sh` to generate through LRP and SVM weights the global relevance mappings of DeepCOMBI and COMBI for all diseases and save them in `FINAL_RESULTS_DIR/deepcombi_raw_rm/` and `FINAL_RESULTS_DIR/combi_raw_rm/`.

- Run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_pipeline.py::TestPipeline::test_save_scaled_averaged_rm` to generate from the raw relevance mappings the scaled (`deepcombi_scaled_rm/`)and filtered (`deepcombi_avg_rm/`) weights.

- To run the permutation test procedure for obtaining chromosomewise significance thresholds of DeepCOMBI and COMBI for all diseases run `qsub -t 1-154 perm_job.sh` 

- To create Figure 4 of the paper run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_wtccc_plots.py::TestWTCCCPlots::test_generate_all_val_accs`.

- To create Figure 5 of the paper run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_wtccc_plots.py::TestWTCCCPlots::test_generate_all_diseases_manhattan_plots`.

- To create Figure 6 of the paper run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_wtccc_plots.py::TestWTCCCPlots::test_generate_roc_recall_curves`.

- To create Table 2 and Table 3 of the paper run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_wtccc_plots.py::TestWTCCCPlots::test_permtestresults` to save the results of the permutation test procedure.
