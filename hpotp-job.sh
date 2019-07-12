#!/bin/bash
#$ -wd /home/hx/PythonImplementation
#$ -o /home/hx/PythonImplementation/qsub
#$ -e /home/hx/PythonImplementation/qsub
#$ -l cuda=1
#$ -t 1-31

source ~/.bashrc
echo "bash executed"

output_path="$PWD/qsub_results/$1/$SGE_TASK_ID.csv"
mkdir $PWD/qsub_results/$1

# Call function that write results sth
ROOT_DIR=$PWD python -m pytest -s tests/test_dnn.py::TestDNN::test_hp_params --output_path $output_path --rep $2


echo "tests ran"
