import math
import numpy as np


def count_lines(filename):
   return sum(1 for line in open(filename))


def count_columns(filename):
    with open(filename) as f:
        first_line = f.readline()
        return math.floor(len(first_line)/3)


def moving_average(x, k, power=1):
        wnew = np.concatenate((np.zeros(0,(k-1)/2),x,np.zeros(0,(k-1)/2-1)))
        wnew = np.cumsum(wnew)
        wnew = np
        #wnew = [zeros(1,(k-1)/2+1), w, zeros(1,(k-1)/2)];
        #wnew = cumsum(wnew);
        # wnew = (wnew(k+1:end)-wnew(1:end-k)).^(1/p);
        # wnew = wnew./(k.^(1/p));
        #cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
