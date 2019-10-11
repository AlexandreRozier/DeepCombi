#!/bin/bash
#$ -wd /home/hx/PythonImplementation
#$ -o /home/hx/PythonImplementation/qsub_output/
#$ -e /home/hx/PythonImplementation/qsub_errors/
source ~/.bashrc

echo "Bootstraping..."

ROOT_DIR=$PWD python -m pytest -s tests/test_pipeline.py::TestPipeline::test_save_combi_rm_and_indices


echo "Over."
