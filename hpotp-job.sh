#!/bin/bash
#$ -wd /home/hx/PythonImplementation
#$ -o /home/hx/PythonImplementation/qsub
#$ -e /home/hx/PythonImplementation/qsub
#$ -j y 
#$ -t 1-31

source ~/.bashrc
echo "bash executed"

output_path="$PWD/qsub_results/$SGE_TASK_ID.csv"

# Call function that write results sth
ROOT_DIR=$PWD python -m pytest -s tests/test_dnn.py::TestDNN::test_hp_params --output_path $output_path 


echo "tests ran"
