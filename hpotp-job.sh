#!/bin/bash
#$ -wd /home/hx/PythonImplementation
#$ -o /home/hx/PythonImplementation/qsub_output/
#$ -e /home/hx/PythonImplementation/qsub_errors/
#$ -l cuda=1
#$ -t 1-31

source ~/.bashrc
echo "Bootstraping..."

output_path="$PWD/hps_results/$1/"

mkdir $PWD/hps_results/$1

# Call function that write results sth
ROOT_DIR=$PWD python -m pytest -s tests/test_cnn.py::TestCNN::test_hp_params --output_path $output_path --rep $2


echo "Over."
