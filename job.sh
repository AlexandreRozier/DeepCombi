#!/bin/bash
#$ -wd $PWD
#$ -o /home/hx/PythonImplementation/qsub_output/
#$ -e /home/hx/PythonImplementation/qsub_errors/
#$ -t 1-5
source ~/.bashrc
echo "bash executed"
ROOT_DIR=$PWD PREFIX=$2 python -m pytest  -s tests/$1
echo "tests ran"
