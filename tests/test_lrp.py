import os

import matplotlib

from combi import chi_square
from helpers import postprocess_weights, plot_pvalues
from parameters_complete import top_k, filter_window_size, p_svm, p_pnorm_filter, IMG_DIR, n_total_snps, TB_DIR

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import MontaezNet, best_params_montaez
from helpers import train_torch_model
    


class TestLRP(object):



    def test_montaez_run(self, fm, labels_0based, indices):
        model = MontaezNet(params=best_params_montaez).cuda()
        train_torch_model(model, fm('2d')['0'][:], labels_0based['0'], indices['0'], best_params_montaez, os.path.join(TB_DIR,'test_m_torch'))

    def test_lrp_explanation(self, fm, labels_0based, indices, h5py_data, labels):
        
        data = torch.Tensor(fm('2d')['0'][:]).cuda()
        data = data.unsqueeze(1)
        labels_0b = torch.LongTensor(labels_0based['0']).cuda()

        g  = dict(
                epochs= 40,
                l1_reg=1.584922,
                l2_reg=2.511921,
                lr=10e-3,
                dropout_rate=0.3,
                batch_size=32
        )


        model = ExConvNet().cuda()
        model, history = train_torch_model(model, data, labels_0b, indices, g)
        
        R = model(data)
        weights = model.relprop(data, R).data.numpy().sum(0)


        top_indices_sorted, _ = postprocess_weights(
            weights, top_k, filter_window_size, p_svm, p_pnorm_filter)
            
        complete_pvalues = chi_square(h5py_data['0'][:], labels['0'])

        fig, axes = plt.subplots(2, sharex='col')
        axes[1].plot(range(n_total_snps),weights.reshape(-1,3).sum(1))
        plot_pvalues(complete_pvalues, top_indices_sorted, axes[0])
        fig.savefig(os.path.join(IMG_DIR,'manhattan-custom-lrp-test.png'))





    