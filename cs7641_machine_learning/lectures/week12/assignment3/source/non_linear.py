# load and preprocess
from wrangle import final_dataset

# manipulate data
import pandas as pd
import numpy as np

# machine learning models
from sklearn.manifold import (Isomap,
                              LocallyLinearEmbedding,
                              SpectralEmbedding,
                              LocallyLinearEmbedding,
                              MDS,
                              TSNE,
                              )

# model evaluation


np.random.seed(123)

def manifold_learning(df:pd.DataFrame):
    pass

def main():

    # CVD
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')

    manifold_learning(df=X_train_scaled_cd)

    # NF
    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')

    manifold_learning(df=X_train_scaled_nf)

if __name__ == "__main__":
    main()