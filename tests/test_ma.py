import numpy as np
from helpers import moving_average

class TestMA(object):
    
    def test_ma(self):
        w = np.arange(10)
        print(w)

        k = 5
        print(moving_average(w,k))
        print("+++++++++++++++")
        
