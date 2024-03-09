# load and preprocess
from wrangle import final_dataset

# manipulate data
import pandas as pd
import numpy as np

# machine learning models
from sklearn.decomposition import (PCA,
                                   KernelPCA,
                                   FastICA)

from sklearn.random_projection import (GaussianRandomProjection,
                                       SparseRandomProjection)

# model evaluation


np.random.seed(123)

def pca(X_train:pd.DataFrame, X_test:pd.DataFrame, kern='General'):
    """_summary_

    Args:
        X_train (pd.DataFrame): _description_
        X_test (pd.DataFrame): _description_
    """

    print('Fitting PCA')
    match kern:
        case 'Kernel':
            print('Fitting and Transforming with KernelPCA')
            transformer = KernelPCA(n_components=7, kernel='linear')
            X_transformed = transformer.fit_transform(X_train)
            X_transformed.shape
            print('Done with KernelPCA')

        case 'General':
            print('Fitting General PCA')
            model = PCA(svd_solver='auto',
                        random_state=123)
            fit_pca = model.fit(X_train)
            print('Done with General PCA')
    print('Done with PCA\n')

    return 1        

def ica(X_train:pd.DataFrame, X_test:pd.DataFrame):
    """_summary_

    Args:
        X_train (pd.DataFrame): _description_
        X_test (pd.DataFrame): _description_
    """

    print('Fitting ICA')
    transformer = FastICA(n_components=7,
                          random_state=123,
                          whiten='unit-variance')
    X_transformed = transformer.fit_transform(X_train)
    X_transformed.shape
    print('Done with ICA\n')

    return X_transformed

def randomized_projections(X_train:pd.DataFrame, X_test:pd.DataFrame, which='Gaussian'):
    """_summary_

    Args:
        X_train (pd.DataFrame): _description_
        X_test (pd.DataFrame): _description_
    """

    print('Fitting and Transforming Randomized Projections')
    match which:
        case 'Gaussian':
            print('Fitting and Transforming with Gaussian')
            transformer = GaussianRandomProjection(n_components=3,
                                                   eps=0.9,
                                                   random_state=123)
            X_new = transformer.fit_transform(X_train)
            X_new.shape
            print('Done with Gaussian')

        case 'Sparse':
            print('Fitting and Transforing with Sparse')
            transformer = SparseRandomProjection(n_components=3,
                                                 eps=0.9,
                                                 random_state=123)
            X_new = transformer.fit_transform(X_train)
            X_new.shape
            print('Done with Sparse')
    print('Done with Randomized Projections\n')

    return X_new            

def main():

    # cvd
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')
    
    # nf
    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')

    # pca
    for k in ['Kernel', 'General']:
        pca(X_train=X_train_scaled_cd, X_test=X_test_scaled_cd, kern=k)
        pca(X_train=X_train_scaled_nf, X_test=X_test_scaled_nf, kern=k)

    # ica
    ica(X_train=X_train_scaled_cd, X_test=X_test_scaled_cd)
    ica(X_train=X_train_scaled_nf, X_test=X_test_scaled_nf)

    # randomized projections
    for w in ['Gaussian', 'Sparse']:
        randomized_projections(X_train=X_train_scaled_cd, X_test=X_test_scaled_cd, which=w)
        randomized_projections(X_train=X_train_scaled_nf, X_test=X_test_scaled_nf, which=w)

if __name__ == "__main__":
    main()