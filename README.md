# DeepCOMBI

This repository contains the implementation of the DeepCOMBI method, 
a neural-network-based improvement of COMBI, a Genome-wide Association Studies tool.  
COMBI is described [here](https://www.nature.com/articles/srep36671#methods).

#### Content:

- `data/`:  Contains synthetic and real genomic data used in DeepCOMBI 
- `img/`:   
- `notebooks/`:   
- `qsub_scripts/`:  
- `misc/`: Small tests used to assess of the performance of various file formats 
- `tests/`:   Pytest-based implementation of DeepCOMBI, allowing to train various models
- `experiments/`:   A folder per experiment on WTCCC data. Each subfolder corresponds to a new NN arcthitecture
and hyperparameters search, ...
 

        

#### Synthetic Data workflow
- Run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_data_generation.py::TestDataGeneration::test_synthetic_genotypes_generation --rep 100` to generate    
`rep` different genotypes and save them into `data/synthetic/genomic.h5py`

- Run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_data_generation.py::TestDataGeneration::test_feature_map_generation` to generate 
the features matrices associated to genomic datasets previously created in `data/synthetic/genomic.h5py`.   
Saves them into `data/synthetic/2d_fm.h5py` and  `data/synthetic/3d_fm.h5py`


#### WTCCC Data workflow

- Run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_data_generation.py::TestDataGeneration::test_mat_preprocessing` 
to transform genomic (N , d x 3) matrices into (N, d, 2) genomic matrices. 
This should be done only once, and then most genomic files can be deleted (except from chromosom 1 and 2 from Crohn's disease)

- Run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_data_generation.py::TestDataGeneration::test_real_pvalues_generation` to 
compute and save for each chromosom of each disease the associated P-Values (saved in `data/WTCCC/pvalues`) 
This should be done only once.

- Set in `parameters_complete.py` FINAL_RESULTS_DIR, which indicates where to write results of this experiment.

- Run `qsub -t 1-7 hpsearch.sh` to run an hyperparameter search amongst all chromosoms of all diseases

- Run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_lotr.py::TestLOTR::test_extract_best_hps` to parse the results 
of the hyperparams search and export them into `FINAL_RESULTS_DIR/hyperparameters`

- Run `qsub -t 1-154 train_models_with_best_params.sh` to train models wth those hyperparameters. 
The `-t 1-154` is used to spawn a job per disease, per chromosom (7*22 = 154)

- Run `qsub -t 1-7 save_deepcombi_rm.sh` and `qsub -t 1-7 save_combi_rm.sh` to generate through LRP the global relevance mappings obtained for the whole 
genome on each disease. Saved in `FINAL_RESULTS_DIR/deepcombi_raw_rm/` and `FINAL_RESULTS_DIR/combi_raw_rm/`

- Run `ROOT_DIR=$PWD SGE_TASK_ID=1 python -m pytest -s tests/test_pipeline.py::TestPipeline::test_save_scaled_averaged_rm` to generate 
from the raw relevance mappings the scaled (`deepcombi_scaled_rm/`)and filtered (`deepcombi_avg_rm/`) weights

- Plot stuff thanks to functions in `test_wtccc_plots.py`  
