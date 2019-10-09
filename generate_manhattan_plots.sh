#!/bin/bash
#$ -wd /home/hx/PythonImplementation
#$ -o /home/hx/PythonImplementation/qsub_output/
#$ -e /home/hx/PythonImplementation/qsub_errors/
source ../pypa-virtualenv-ce9343c/combi/bin/activate # Necessary for model training

echo "Bootstraping..."

ROOT_DIR=$PWD python -m pytest -s tests/test_lotr.py::TestLOTR::test_generate_plots


echo "Over."
