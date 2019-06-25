IDEAs:  regulazrization to force locality ?


# 7-12 / 04

Python implementation of Combi
added feature matrix extraction
added svm training
added custom moving average

# 14-19 / 04

Learned Chi2 test
Started writing report: chi2 started & mathematical overview of the topic
Added unit testing
Implemented permutations for t* computations

# 22 - 26 / 04

Added LRP through innvestigate
- DeepTaylor + 500 dense relu & softmax works well, but the relevance map is absolute noise
- need simpler model & more data (MNISt works but n=60k & d=900)

# 29/04 - 03 / 05

- Hp optimization with talos
- Trying dense, conv, dropout to get good generalization, in hope to improve LRP. Conv should enforce locality
- Read LRP paper thoroughly, asked for more dataaaa (access to WTCCC to get better results)
- shitty result w/ linear model (<40% validation accuracy %) (Dense2)
- convolutional model performs quite well (100% val acc with conv1 1 filter size 35; avgpool size 5, dense2)

# 06 - 10 / 05

- Solved bugs with custom Keras constraints
- Working on toy data generation thanks to chromosom1 & 2 from Bettina

# 13 - 17 / 05

- Built streaming version of the data (generators)
- Now Accuracy has dropped to 60%
- qsub -l cuda=1 -wd $PWD -o $PWD/qsub -j y test.sh   //Tesla P100-PCIE-16GB works

# 20 - 24 / 05

- Trying different HP with tensorboard and QSUB
- Max accuracy : 65%. 
- Dropout works well (> 0.3)
- Few epochs with lr 0.01
- Higher batch_size (> 64)
- Lots of epochs with lr 10e-4
- best: 600 batch size, 0.5 dropout, momentum 0.1, lr 10e-4

# 27 - 31 / 05

- Corrected sheet4
- max train acc: 80%
- max val acc: 65%
    -   So need better arch
- PCA as preprocessing step ?
- Read papers on DNN 
- Tried VGG, dense, categorical vs non categorical loss
- Sigmoid on dense NN, relu on deep
- **Idea:** Try l1 on dense network to promote sparsity -> **+6% val accuracy!!! Some models reach 72% val_acc**
- **Idea:** Conv networks with huge window size + max pool to find peak (conv extract)

# 03 - 07 /  05

- Discussion with nico, avoid batchnorm that reduces to gaussian our data distribution
- Avoid mean centering when it's not for SVM
- Baselines on 10 000 syn data:  

| Baseline      | Max train acc | Max val acc           | AUC Roc         |
| ------------- |:-------------:| -----:                |-----:         |
| dense         | 1.0           | 0.65                  |
| svm + l2      | ?             |   0.63                |   0.67 |  Not implemented |
| decision trees | ?            |   0.6737373737373737  |   0.6737373737373737 |
| logistic regression + l2| ?   |     0.6595            |   0.6629 |
  
  **FEATURE MATRIX FOR COMBI NEEDS TO BE SCALED**

# 10 - 14 /  05

- Fixed data generation once for all
- Tried SVM on only the 20 informative SNPs:

| N_subjects    | SVM val_acc       | C                     |
| ------------- |:-------------:    |:--:                   |
| 300           | 0.83              |C=0.00091              |

| N_subjects    | Kernel SVM val_acc        | C             |
| ------------- |:-------------:            |:--:           |
| 300           | 0.83                      |C=1            |

N_subject= 4000

| Baseline    |   train_acc | val_acc        | C             |
| ------------- |---------- |:-------------:            |:--:           |
| svm          | 0.76 |0.76                    |C=0.077        |
| kernel svm      |?   | 0.76                    |C=0.715        |
| decision trees  |?        | 0.71                   | NA      |
| linear          |?| 0.74                   |NA        |

## Test of the ttbr parameter (tower to base ratio)
Mit 4000 subjects, 10 000 noise_snps, 20 inform_snps, 50 different C-values:

- ttbr = 6:    gibt    svm train_acc 1.0 +/- 0.2, val_acc 0.665  + wählt snps, die zur Tower gehören  

- ttbr = 1:    gibt    svm train_acc 0.85 +/- 0.2, val_acc 0.53 + kann kein snps wählen, die zur Tower gehören

>So die ergebnisse machen total Sinn   

**Aber** mit ttbr =1 , die Tower existiert noch (~ log-pvalue = 80 im vergleich mit den anderen, die alle 0 sind), **trotz die SVM die nicht erkennen kann**.