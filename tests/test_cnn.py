import numpy as np
import pandas as pd
import os 
import time

import tensorflow


import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.optimizers import SGD
from models import create_explanable_conv_model, create_explanable_conv_model, create_lenet_model, create_clairvoyante_model, create_montaez_dense_model, create_montaez_dense_model
from models import ConvDenseRLRonP
from tqdm import tqdm
from combi import combi_method
from helpers import postprocess_weights, chi_square, compute_metrics, plot_pvalues, generate_name_from_params, generate_syn_phenotypes, simple_postprocess_weights

from parameters_complete import random_state, nb_of_nodes, pnorm_feature_scaling, filter_window_size, p_svm, p_pnorm_filter, n_total_snps, top_k, ttbr, thresholds, IMG_DIR, DATA_DIR
from joblib import Parallel, delayed
from combi import classifier
import innvestigate


best_params = {
    'epochs': 500,
    'batch_size': 32,   
    'l1_reg': 1e-4,
    'l2_reg': 1e-6,
    'lr' : 0.01,
    'dropout_rate':0.3,
    'factor':0.4,
    'patience':100,
}

class TestCNN(object):

    def test_indices(self, labels_0based, indices):
        idx = indices['0']
        test_labels = labels_0based['0'][idx.test]
        train_labels = labels_0based['0'][idx.train]
        print(test_labels)
        print(train_labels)
        print(len(test_labels[test_labels == 1]))
        print(len(test_labels[test_labels == 0]))
        print(len(train_labels[train_labels == 1]))
        print(len(train_labels[train_labels == 0]))

    def test_train(self, fm, labels_0based, indices, rep, output_path):

        fm_ = fm('3d')        
        
        def f(x,y,idx,key):
            with tensorflow.Session().as_default():
                model = create_montaez_dense_model(best_params)
                model.fit(x=x[idx.train],
                        y=y[idx.train],
                        validation_data=(x[idx.test], y[idx.test]),
                        epochs=best_params['epochs'],
                        callbacks=[
                            ReduceLROnPlateau(monitor='val_loss', 
                                factor=best_params['factor'], 
                                patience=best_params['patience'], 
                                mode='min'),                            
                            TensorBoard(log_dir=os.path.join(output_path,'tb','test'+key),
                                histogram_freq=3,
                                write_graph=False,
                                write_grads=True,
                                write_images=False,
                            )
                        ],
                        verbose=1)
            
        Parallel(n_jobs=-1, prefer="threads")(delayed(f)(
                                                fm_[str(i)][:],
                                                labels_0based[str(i)],
                                                indices[str(i)],
                                                str(i)
                                            ) for i in range(rep))


    def test_conv_lrp(self, h5py_data, labels, fm, labels_0based, indices,rep, tmp_path):
        
        fig, axes = plt.subplots(6,2*rep, squeeze=True)
        fig.set_size_inches(30, 30)

        def f(i, x_3d, x_2d, y, y_0b, idx):
            with tensorflow.Session().as_default():
                model = create_montaez_dense_model(best_params)
                model.fit(x=x_3d[idx.train],
                    y=y_0b[idx.train],
                    validation_data=(x_3d[idx.test], y_0b[idx.test]),
                    epochs=best_params['epochs'],
                    callbacks=[
                        ReduceLROnPlateau(monitor='val_loss', 
                                    factor=best_params['factor'], 
                                    patience=best_params['patience'], 
                                    mode='min'),   
                        ],
                verbose=1)
            

                classifier.fit(x_2d, y)
                svm_weights = classifier.coef_[0] # n_snps * 3
                axes[0][2*i+1].plot(np.absolute(svm_weights).reshape(-1,3).sum(1))
                axes[1][2*i+1].plot(np.absolute(svm_weights).reshape(-1,3).sum(1))
                axes[2][2*i+1].plot(np.absolute(svm_weights).reshape(-1,3).sum(1))
                axes[3][2*i+1].plot(np.absolute(svm_weights).reshape(-1,3).sum(1))
                axes[4][2*i+1].plot(np.absolute(svm_weights).reshape(-1,3).sum(1))
                axes[5][2*i+1].plot(np.absolute(svm_weights).reshape(-1,3).sum(1))

                # LRPZ
                analyzer = innvestigate.analyzer.LRPZ(model)
                weights = analyzer.analyze(x_3d[y_0b>0.9]).sum(0)
                axes[0][2*i].plot(np.absolute(weights).reshape(-1,3).sum(1), label='ttbr={},lrpz'.format(ttbr))
                
                
                # LRPEpsilon
                analyzer = innvestigate.analyzer.LRPEpsilon(model, epsilon=1e-5)
                weights = analyzer.analyze(x_3d[y_0b>0.9]).sum(0) 
                axes[1][2*i].plot(np.absolute(weights).reshape(-1,3).sum(1), label='epsilon')

                # LRPAlpha1Beta0
                analyzer = innvestigate.analyzer.LRPAlpha1Beta0(model)
                weights = analyzer.analyze(x_3d[y_0b>0.9]).sum(0)
                axes[2][2*i].plot(np.absolute(weights).reshape(-1,3).sum(1), label='a-1, b-0')


                # LRPAlpha2Beta1
                analyzer = innvestigate.analyzer.LRPAlpha2Beta1(model)
                weights = analyzer.analyze(x_3d[y_0b>0.9]).sum(0)
                axes[3][2*i].plot(np.absolute(weights).reshape(-1,3).sum(1),label='a-2, b-1')

                # LRPZplus
                analyzer = innvestigate.analyzer.LRPZPlus(model)
                weights = analyzer.analyze(x_3d[y_0b>0.9]).sum(0)
                axes[4][2*i].plot(np.absolute(weights).reshape(-1,3).sum(1),label='lrpzplus')

                # LRPAlpha1Beta0IgnoreBias
                analyzer = innvestigate.analyzer.LRPAlpha1Beta0IgnoreBias(model)
                weights = analyzer.analyze(x_3d[y_0b>0.9]).sum(0)
                axes[5][2*i].plot(np.absolute(weights).reshape(-1,3).sum(1),label='LRPAlpha1Beta0IgnoreBias')

        Parallel(n_jobs=-1, prefer="threads")(delayed(f)(i, fm("3d")[str(i)][:], fm("2d")[str(i)][:], labels[str(i)], labels_0based[str(i)], indices[str(i)]) for i in range(rep))
            
        fig.savefig(os.path.join(IMG_DIR, 'montaez-lrp.png'))


    def test_hp_params(self, fm, labels_0based, indices, rep, output_path):
        fm = fm('3d')
        
        datasets = [fm[str(i)][:] for i in range(rep)]

        params_space = {
            'epochs': [1000],
            'batch_size': [32],
            'l1_reg': np.logspace(-4, -4, 1),
            'l2_reg': np.logspace(-6, -6, 1),
            'lr' : np.logspace(-2, -2, 1),
            'dropout_rate':np.linspace(0.1,0.5,3),
            'patience':np.linspace(30,100,5),
            'factor':np.linspace(0.1,0.5,5),

        }

        grid = ParameterGrid(params_space)
        BUDGET = 100
        grid = random_state.choice(list(grid), BUDGET)
        
        print("TESTING {} PARAMETERS".format(len(grid)))
        params_array_per_node = np.array_split(grid, nb_of_nodes)

        def f(g):

            time.sleep(0.1)
            name = generate_name_from_params(g)
            with tensorflow.Session().as_default():
                model = create_montaez_dense_model(g)

                histories = [model.fit(x=fm[indices[str(i)].train],
                                        y=labels_0based[str(i)][indices[str(i)].train],
                                        validation_data=(
                                           fm[indices[str(i)].test], labels_0based[str(i)][indices[str(i)].test]),
                                        epochs=g['epochs'],
                                        callbacks=[
                                            ReduceLROnPlateau(monitor='val_loss', 
                                                factor=best_params['factor'], 
                                                patience=best_params['patience'], 
                                                mode='min'),   
                                            TensorBoard(log_dir=os.path.join(output_path,'tb',name+str(i)),
                                                histogram_freq=3,
                                                write_graph=False,
                                                write_grads=True,
                                                write_images=False,
                                            )
                                        ],
                                        verbose=1).history for i, fm in enumerate(datasets)]
                mean_val_loss = np.mean([history['val_loss'][-1] for history in histories])
                for history in histories:
                    history['mean_val_loss'] = mean_val_loss

            return [{**g, **history} for history in histories]

        hparams_array = params_array_per_node[int(os.environ['SGE_TASK_ID'])-1]

        entries = np.array(Parallel(n_jobs=-1, prefer="threads")
                           (delayed(f)(g) for g in hparams_array)).flatten()
        results = pd.DataFrame(list(entries))
        results.to_csv(os.path.join(output_path, os.environ['SGE_TASK_ID']+'.csv'))


    def test_svm_cnn_comparison(self, fm, labels, labels_0based, rep, indices):
        """ Compares performance of SVM and CNN models
        """
        
        fm_3d = fm("3d")
        fm_2d = fm("2d")

        def fit_cnn(x, y, idx):
            model = create_montaez_dense_model(best_params)
            return model.fit(x=x[idx.train],
                            y=y[idx.train],
                            validation_data=(
                                x[idx.test], y[idx.test]),
                            epochs=best_params['epochs'],
                            callbacks=[
                                ReduceLROnPlateau(monitor='val_loss', 
                                    factor=best_params['factor'], 
                                    patience=best_params['patience'], 
                                    mode='min'),   
                        ]).history['val_acc'][-1]

        def fit_svm(x, y, idx):
            svm_model = classifier.fit(x[idx.train], y[idx.train])
            return svm_model.score(x[idx.test], y[idx.test])
            

        cnn_val_acc = Parallel(
            n_jobs=30)(delayed(fit_cnn)(fm_3d[str(i)][:], labels_0based[str(i)], indices[str(i)]) for i in tqdm(range(1, rep)))
        svm_val_acc = Parallel(
            n_jobs=30)(delayed(fit_svm)(fm_2d[str(i)][:], labels[str(i)], indices[str(i)]) for i in tqdm(range(1, rep)))

        cnn_val_acc = cnn_val_acc * 100
        svm_val_acc = svm_val_acc * 100

        # 3. Compare average val_acc on all syn datasets
        print('SVM val_acc mean {}/std{}/max{}/min{}; cnn val_acc mean {}/std{}/max{}/min{}'.format(
            np.mean(svm_val_acc),np.std(svm_val_acc), np.max(svm_val_acc),np.min(svm_val_acc),
            np.mean(cnn_val_acc),np.std(cnn_val_acc), np.max(cnn_val_acc),np.min(cnn_val_acc)))




    def test_tpr_fwer(self, h5py_data, labels, labels_0based, fm, indices, rep, true_pvalues):
        """ Compares combi vs dense curves
        """

        window_lengths = [11, 21, 31, 35, 41, 51, 61, 101]

        fig, axes = plt.subplots(2)
        fig.set_size_inches(18.5, 10.5)
        ax1, ax2 = axes

        ax1.set_ylabel('TPR')
        ax1.set_xlabel('FWER')
        ax1.set_ylim(0, 0.45)
        ax1.set_xlim(0, 0.1)

        ax2.set_ylabel('Precision')
        ax2.set_xlabel('TPR')

        def combi_compute_pvalues(d, x, l):

            indices, pvalues = combi_method(d, x, l, pnorm_feature_scaling,
                                            filter_window_size, top_k)
            pvalues_filled = np.ones(n_total_snps)
            pvalues_filled[indices] = pvalues
            del d, l
            return pvalues_filled

        def challenger_compute_pvalues(d, x, l_0b, l, idx):
            isOnlyZeros = False
            with tensorflow.Session().as_default():

                model_wo_sm = create_montaez_dense_model(best_params)

                model_wo_sm.fit(x=x[idx.train], y=l_0b[idx.train],
                                validation_data=(x[idx.test], l_0b[idx.test]),
                                epochs=best_params['epochs'],
                                callbacks=[
                                    ReduceLROnPlateau(monitor='val_loss', 
                                        factor=best_params['factor'], 
                                        patience=best_params['patience'], 
                                        mode='min'),   
                                ])

                analyzer = innvestigate.analyzer.LRPAlpha1Beta0(model_wo_sm)
                weights = analyzer.analyze(x[l_0b>0.9]).sum(0)

                
                if np.max(abs(weights)) < 0.005:
                    fig, axes = plt.subplots(1)
                    isOnlyZeros = True
                    axes.plot(np.absolute(weights).sum(axis=1))
                    fig.savefig(os.path.join(IMG_DIR, 'test.png'))
                
                pvalues_list = np.zeros((len(window_lengths), weights.shape[0]))
                for i, filter_size in enumerate(window_lengths):
                    top_indices_sorted,_ = postprocess_weights(
                        weights, top_k, filter_size, p_svm, p_pnorm_filter)
                    pvalues = chi_square(d[:, top_indices_sorted], l)
                    pvalues_filled = np.ones(n_total_snps)
                    pvalues_filled[top_indices_sorted] = pvalues
                    pvalues_list[i] = pvalues_filled
                del d, x, l

            return pvalues_list, isOnlyZeros

        fm_2d = fm("2d")
        fm_3d = fm("3d")

        pvalues_per_run_combi = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(
            combi_compute_pvalues)(h5py_data[str(i)][:], fm_2d[str(i)][:], labels[str(i)]) for i in tqdm(range(rep))))

        # len(thresholds) * len(window_sizes) * 10020
        a = Parallel(n_jobs=-1, require='sharedmem')(delayed(
            challenger_compute_pvalues)(h5py_data[str(i)][:], fm_3d[str(i)][:], labels_0based[str(i)], labels[str(i)], indices[str(i)]) for i in tqdm(range(rep)))


        # INNvestigate bugfix
        zeros_index = np.array(list(np.array(a)[:,1]))
        pvalues_per_run_dense = np.array(list(np.array(a)[:,0]))

        pvalues_per_run_combi = pvalues_per_run_combi[np.logical_not(zeros_index)]
        pvalues_per_run_dense = pvalues_per_run_dense[np.logical_not(zeros_index)]
        true_pvalues = true_pvalues[np.logical_not(zeros_index)]
        
        res_combi = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(
            pvalues_per_run_combi, true_pvalues, threshold) for threshold in tqdm(thresholds)))

        tpr_combi, _, fwer_combi, precision_combi = res_combi.T
        
        ax1.plot(fwer_combi, tpr_combi, '-o',
                label='Combi - ttbr={}'.format(ttbr))
        ax2.plot(tpr_combi, precision_combi, '-o',
                label='Combi  - ttbr={}'.format(ttbr))
        
        for i,window in enumerate(window_lengths):
            pvalues_challenger = pvalues_per_run_dense[:,i]


            res_dense = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(compute_metrics)(
                pvalues_challenger, true_pvalues, threshold) for threshold in tqdm(thresholds)))

            tpr_dense, _, fwer_dense, precision_dense = res_dense.T

            assert fwer_combi.max() <= 1 and fwer_combi.min() >= 0
            ax1.plot(fwer_dense, tpr_dense, '-x',
                    label='Challenger  - k={}, ttbr={}'.format(window, ttbr))

            ax2.plot(tpr_dense, precision_dense, '-x',
                    label='Challenger - k={}, ttbr={}'.format(window, ttbr))

        ax1.legend()
        ax2.legend()
        fig.savefig(os.path.join(IMG_DIR, 'tpr_fwer_montaez_k_a1b0-bugfix.png'), dpi=300)
        print("FOUND {} BUGS!".format(zeros_index.sum()))



    def test_lrp_svm(self, h5py_data, fm, indices, rep, tmp_path):
        """ Compares efficiency of the combi method with several TTBR
        """
        ttbrs = [20, 6, 1, 0]
        h5py_data = h5py_data['4'][:]
        idx = indices['4']
        fig, axes = plt.subplots(len(ttbrs), 4, sharex='col')
        x_3d = fm("3d")['0'][:]
        x_2d = fm("2d")['0'][:]

        for i, ttbr in enumerate(ttbrs):
            print('Using tbrr={}'.format(ttbr))
            labels = generate_syn_phenotypes(ttbr=ttbr, quantity=rep)['4']
            l_0b = (labels+1)/2
           
            model = create_montaez_dense_model(best_params)
            model.fit(x=x_3d[idx.train],
                    y=l_0b[idx.train],
                    validation_data=(x_3d[idx.test], l_0b[idx.test]),
                    epochs=best_params['epochs'],
                    callbacks=[
                        ReduceLROnPlateau(monitor='val_loss', 
                                factor=best_params['factor'], 
                                patience=best_params['patience'], 
                                mode='min'),                   
                    ],
            )

            analyzer = innvestigate.analyzer.LRPZ(model)
            weights = analyzer.analyze(x_3d[l_0b>0.9]).sum(0) # sum over sick ppl
    
            top_indices_sorted, postprocessed_weights = postprocess_weights(
                weights, top_k, filter_window_size, p_svm, p_pnorm_filter)

            complete_pvalues = chi_square(h5py_data, labels)
            plot_pvalues(complete_pvalues, top_indices_sorted, axes[i][0])
            
            # Plot distribution of relevance
            axes[i][1].plot(np.absolute(weights).reshape(-1,3).sum(1), label='ttbr={}'.format(ttbr))
            axes[i][1].legend()
            axes[i][1].set_title('Absolute relevance')

            # Plot distribution of postprocessed vectors
            axes[i][2].plot(postprocessed_weights, label='ttbr={}'.format(ttbr))
            axes[i][2].legend()
            axes[i][2].set_title('Postprocessed relevance')


            # Plot distribution of svm weights
            svm_weights = classifier.fit(x_2d, labels).coef_
            axes[i][3].plot(np.absolute(svm_weights).reshape(-1,3).sum(1), label='ttbr={}'.format(ttbr))
            axes[i][3].legend()
            axes[i][3].set_title('Absolute SVM weight')

        
            fig.savefig(os.path.join(IMG_DIR, 'manhattan-montaez-test.png'))