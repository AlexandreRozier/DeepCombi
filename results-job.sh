#!/bin/bash
#$ -wd /home/hx/PythonImplementation
#$ -o /home/hx/PythonImplementation/qsub_output/
#$ -e /home/hx/PythonImplementation/qsub_errors/
#$ -l cuda=1
#$ -t 1

source ~/.bashrc
echo "Bootstraping..."

# Call function that write results sth
ROOT_DIR=$PWD python -m pytest -s tests/test_cnn.py::TestCNN::test_tpr_fwer --rep $1


echo "Over."
