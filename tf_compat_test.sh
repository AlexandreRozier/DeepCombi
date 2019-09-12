#!/bin/bash
#$ -wd $PWD
#$ -t 1-100
#$ -o /home/hx/PythonImplementation/qsub_output/
#$ -e /home/hx/PythonImplementation/qsub_errors/
source ~/.bashrc


trap fd SIGCHLD
set -o monitor
fd () {
    if [ $? = 0 ]
    then
        hostname >> compatible_nodes.txt    
        echo 'Sieg! '
    else
        echo 'oh no :/'
    fi
}

python tf_compat_test.py

