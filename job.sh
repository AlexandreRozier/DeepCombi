#!/bin/bash
source ~/.bashrc
echo "bash executed"
ROOT_DIR=$PWD PREFIX=$2 python -m pytest  -s tests/$1
echo "tests ran"
