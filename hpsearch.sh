#!/bin/bash
#$ -wd /home/hx/PythonImplementation
#$ -o /home/hx/PythonImplementation/qsub_output/
#$ -e /home/hx/PythonImplementation/qsub_errors/

source ~/.bashrc
echo "Bootstraping..."

ROOT_DIR=$PWD python -m pytest -s tests/test_lotr.py::TestLOTR::test_hpsearch


echo "Over."
