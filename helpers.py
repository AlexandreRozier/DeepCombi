import math
import numpy as np


def count_lines(filename):
   return sum(1 for line in open(filename))


def count_columns(filename):
    with open(filename) as f:
        first_line = f.readline()
        return math.floor(len(first_line)/3)


def moving_average(w, k, power=1):
        """
        Inspired from https://uk.mathworks.com/matlabcentral/fileexchange/12276-moving_average-v3-1-mar-2008
        """
        assert(k%2==1)
        
        wnew = np.absolute(w)
        wnew = np.power(wnew,power)
        wnew = np.concatenate((
                np.zeros(int((k-1)/2+1)),
                wnew, 
                np.zeros(int((k-1)/2))),
                axis=None)
        wnew = np.cumsum(wnew) 
        assert(wnew[k:].shape==wnew[0:-k].shape)  
        wnew = np.subtract(wnew[k:],wnew[0:-k])
        wnew = np.power(wnew,1.0/power)
        wnew = np.divide(wnew, k**(1.0/power))
        return wnew

# TODO test this
def other_moving_avg(x,k,p):
        x = np.absolute(x)
        x = np.power(x, p)
        d = len(x)
        result = np.zeros(d)
        for j in range(d):
                acc = 0
                for l in range(max(0,j-int((k-1)/2)), min(d,j+int((k-1)/2))):
                        acc += x[l]
                result[j] = acc
        return np.power(result, 1.0/p)