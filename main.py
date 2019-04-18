import numpy as np
from combi import combi_method
from parameters_complete import *
from helpers import moving_average, other_moving_avg

with open('PythonImplementation/data/data.txt', 'r') as d, open('PythonImplementation/data/labels.txt', 'r') as l:
    data = np.loadtxt(d, np.chararray, skiprows=0)
    labels = np.loadtxt(l, dtype=np.int8, skiprows=0)
    combi_method(data, labels, pnorm_feature_scaling, svm_rep,
                 Cs, 2, classy, filter_window_size, p_pnorm_filter, 30)
    print("Pouet")
