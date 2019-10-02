import pandas as pd

from helpers import chi_square


class TestPipeline(object):

    def test_read_data(self, real_h5py_data, real_labels):
        queries = pd.read_csv('../data/queries/BD.txt')
        results = pd.read_csv('../data/queries/BD.txt')

        pvalues = []
        for chromo in range(1,23):
            candidates = chi_square(real_h5py_data('BD',chromo)[:], real_labels('BD'))
            candidates = candidates[candidates <=1e-4]
            pvalues.append(candidates)
        assert len(pvalues) == queries.shape[0]






