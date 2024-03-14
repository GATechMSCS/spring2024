# load and preprocess
from wrangle import final_dataset

# manipulate data
import pandas as pd
import numpy as np

# Baseline Model
from steps import step5

np.random.seed(123)

def put_it_all_together(X_train:pd.DataFrame,
                        y_train:pd.Series,
                        X_test:pd.DataFrame,
                        y_test:pd.Series,
                        dset:str):

    results = {dset: {'step1': {'gm': None,
                                'kmeans': None},
                      'step2': {'pca': None,
                                'ica': None,
                                'sparseRP': None,
                                'manifold': None},
                      'step3': {'pca': {'gm': None ,
                                        'kmeans': None},
                                'ica': {'gm': None,
                                        'kmean': None},
                                'sparseRP': {'gm': None,
                                             'kmeans': None},
                                'manifold': {'gm': None,
                                             'kmeans': None}},
                      'step4': {'pca': None ,
                                'ica': None,
                                'sparseRP': None,
                                'manifold': None},
                      'step5': {'gm': None,
                                'kmean': None}}}

    param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (250,)],  # size of hidden layers
    'activation': ['relu', 'logistic'],  # activation functions
    'alpha': [0.0001, 0.001, 0.01],  # L2 penalty (regularization term)
    'learning_rate_init': [0.001, 0.01, 0.1],  # initial learning rate
    'max_iter': [200, 350, 500], # maximum number of iterations
    'batch_size': [150, 200, 250],} # size of minibatches for stochastic optimizers

    print('\nRunning All Steps')

    results = step5(X_train,
                    y_train,
                    X_test,
                    y_test,
                    dset,
                    results,
                    param_grid)

    print('Completed All Steps')

    return results

def main():

    # CVD 
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')

    # Run CVD Model
    results_cv = put_it_all_together(X_train=X_train_scaled_cd,
                                     y_train=y_train_cd,
                                     X_test=X_test_scaled_cd,
                                     y_test=y_test_cd,
                                     dset='cvd')

    # NF
    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')

    # Run NF Model
    results_nf = put_it_all_together(X_train=X_train_scaled_nf,
                                     y_train=y_train_nf,
                                     X_test=X_test_scaled_nf, 
                                     y_test=y_test_nf,
                                     dset='nf')

    print(results_cv)
    print()
    print(results_nf)

if __name__ == "__main__":
    main()