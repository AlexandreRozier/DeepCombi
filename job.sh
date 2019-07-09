#!/bin/bash
#$ -wd $PWD
#$ -l cuda=1
source ~/.bashrc
echo "bash executed"
ROOT_DIR=$PWD PREFIX=$2 python -m pytest  -s tests/$1
echo "tests ran"
