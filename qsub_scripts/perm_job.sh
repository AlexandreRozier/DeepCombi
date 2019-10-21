#!/bin/bash
#$ -wd /home/hx/PythonImplementation
#$ -o /home/hx/PythonImplementation/qsub_output/
#$ -e /home/hx/PythonImplementation/qsub_errors/
#$ -l cuda=1
#$ -t 1-23

source ~/.bashrc
echo "Bootstraping..."

ROOT_DIR=$PWD python -m pytest -s tests/test_lotr.py::TestLOTR::test_permutations


echo "Over."
