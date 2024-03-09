# load and preprocess
from wrangle import final_dataset

# manipulate data
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
    pass

if __name__ == "__main__":
    main()