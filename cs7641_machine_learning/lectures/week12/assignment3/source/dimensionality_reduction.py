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
                                       SparseRandomProjection,
                                       johnson_lindenstrauss_min_dim)

# model evaluation
# See Below Functions

def pca(df:pd.DataFrame):
    pass

def ica(df:pd.DataFrame):
    pass

def randomized_projections(df:pd.DataFrame):
    pass

def main():

    # CVD
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')

    pca(df=X_train_scaled_cd)
    ica(df=X_train_scaled_cd)
    randomized_projections(df=X_train_scaled_cd)

    # NF
    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')

    pca(df=X_train_scaled_nf)
    ica(df=X_train_scaled_nf)
    randomized_projections(df=X_train_scaled_nf)

if __name__ == "__main__":
    main()