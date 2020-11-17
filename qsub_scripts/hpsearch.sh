#!/bin/bash
#$ -wd /home/bmieth/DeepCombi
#$ -o /home/bmieth/DeepCombi/qsub_output/
#$ -e /home/bmieth/DeepCombi/qsub_errors/

source ~/.bashrc
echo "Bootstraping..."

ROOT_DIR=$PWD python -m pytest -s tests/test_lotr.py::TestLOTR::test_hpsearch_with_weights


echo "Over."
