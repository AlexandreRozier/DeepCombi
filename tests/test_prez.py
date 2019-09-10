import numpy as np
import os 
from combi import classifier
import matplotlib.pyplot as plt
from parameters_complete import IMG_DIR
import tensorflow
import innvestigate
import innvestigate.utils as iutils
from joblib import Parallel, delayed
from keras.callbacks import ReduceLROnPlateau
from models import  create_montaez_dense_model_2
from parameters_complete import ttbr

# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


class TestPrez(object):

    def test_plot_svm(self, fm, labels):
        x_2d = fm('2d')['0'][:]

        fig, axes = plt.subplots(1)
            

        classifier.fit(x_2d, labels['0'])
        svm_weights = classifier.coef_[0] # n_snps * 3
        abs_weights = np.absolute(svm_weights).reshape(-1,3).sum(1)
        abs_weights = abs_weights/np.max(abs_weights)

        axes.plot(abs_weights, color='#104BA9')
        axes.axhline(y=0.6, color='#D9005B')
        axes.set_xlabel('SNP Position')
        axes.set_ylabel('Associated P-Value')
        axes.set_title('Using SVM Weight vector as a Relevance Map')
        

        fig.savefig(os.path.join(IMG_DIR, 'svm.pdf'))

        abs_weights[abs_weights <= 0.6] = 0
        axes.lines[0].remove()
        axes.plot(abs_weights, color='#104BA9')
        fig.savefig(os.path.join(IMG_DIR, 'curated-svm.pdf'))



    
    def test_compare_lrp_svm(self,h5py_data, labels, fm, labels_0based, labels_cat, indices,rep, tmp_path):
        best_params_montaez_2 = {
   
            'epochs': 500,
            'batch_size': 32,   
            'l1_reg': 1e-4,
            'l2_reg': 1e-6,
            'lr' : 0.01,
            'dropout_rate':0.5,
            'factor':0.71,
            'patience':100,
        }
        fig, axes = plt.subplots(2,2, squeeze=True)
        fig.suptitle('Relevance Distribution from SVM and LRP Preprocessing ', fontsize=16)
        x_3d = fm("3d")['0'][:]
        x_2d = fm("2d")['0'][:]
        y = labels['0']
        y_0b = labels_cat['0']
        idx = indices['0']

        model = create_montaez_dense_model_2(best_params_montaez_2)
        model.fit(x=x_3d[idx.train],
            y=y_0b[idx.train],
            validation_data=(x_3d[idx.test], y_0b[idx.test]),
            epochs=best_params_montaez_2['epochs'],
            callbacks=[
                
                ReduceLROnPlateau(monitor='val_loss', 
                            factor=best_params_montaez_2['factor'], 
                            patience=best_params_montaez_2['patience'], 
                            mode='min'),   
                ],
        verbose=1)
        model = iutils.keras.graph.model_wo_softmax(model)

    

        classifier.fit(x_2d, y)
        svm_weights = np.absolute(classifier.coef_[0]).reshape(-1,3).sum(1) # n_snps * 3
        
        axes[0][1].plot(svm_weights)
        axes[0][1].set_title('SVM Preprocessing')

        axes[1][1].plot(svm_weights)
        axes[1][1].set_xlabel('SNP Locus')

        # LRPZ
        axes[0][0].set_title('LRP Preprocessing')

        analyzer = innvestigate.analyzer.LRPZ(model)
        abs_weights = np.absolute(analyzer.analyze(x_3d).sum(0)).sum(1)
        abs_weights = abs_weights/abs_weights.sum()
        axes[0][0].plot(abs_weights, label='LRP Rule: LRP-Z')
        axes[0][0].legend()

        

        # LRPAlpha1Beta0
        analyzer = innvestigate.analyzer.LRPAlpha2Beta1(model)
        abs_weights = np.absolute(analyzer.analyze(x_3d).sum(0)).sum(1)
        abs_weights = abs_weights/abs_weights.sum()
        axes[1][0].plot(abs_weights, label='LRP Rule: LRP-Alpha2Beta1')
        axes[1][0].set_xlabel('SNP Locus')
        axes[1][0].legend()

            
        
        fig.savefig(os.path.join(IMG_DIR, 'svm-lrp-test.pdf'))

        


                 
