# load and preprocess
from wrangle import final_dataset

# manipulate data
import pandas as pd
import numpy as np

# machine learning models
from sklearn.manifold import (Isomap,
                              LocallyLinearEmbedding,
                              SpectralEmbedding,
                              TSNE)

# model evaluation


np.random.seed(123)

def manifold_learning(X_train:pd.DataFrame, X_test:pd.DataFrame, which='iso'):
    """_summary_

    Args:
        X_train (pd.DataFrame): _description_
        X_test (pd.DataFrame): _description_
    """

    print('Fitting and Transforming Manifold Learning')
    match which:
        case 'iso':
            print('Fitting and Transforming with Isomap')
            embedding = Isomap(n_components=2,
                                 n_jobs=-1,
                                 n_neighbors=25)
            X_transformed = embedding.fit_transform(X_train)
            X_transformed.shape
            print('Done with Isomap')

        case 'se':
            print('Fitting and Transforming with Spectral Embedding')
            embedding = SpectralEmbedding(n_components=2,
                                          random_state=123,
                                          n_jobs=-1)
            X_transformed = embedding.fit_transform(X_train)
            X_transformed.shape
            print('Done with Spectral Embedding')

        case 'tsne':
            print('embedding and Transforming with TSNE')
            embedding = TSNE(n_components=2,
                              learning_rate='auto',
                              init='random',
                              perplexity=3,
                              random_state=123,
                              n_jobs=-1)
            X_embedded = embedding.fit_transform(X_train)
            X_embedded.shape
            print('Done with TSNE')

        case 'lle':
            print('Fitting and Transforming with Locally Linear Embedding')
            embedding = LocallyLinearEmbedding(n_components=2,
                                               random_state=123,
                                               n_jobs=-1)
            X_transformed = embedding.fit_transform(X_train)
            X_transformed.shape
            print('Done with Locally Linear Embedding')
    print('Done with Manifold Learning\n')

def main():

    # CVD
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')

    # NF
    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')

    # manifold learning
    for w in ['iso', 'se', 'tsne', 'lle']:
        manifold_learning(X_train=X_train_scaled_cd,
                      X_test=X_test_scaled_cd,
                      which=w)
        manifold_learning(X_train=X_train_scaled_nf,
                      X_test=X_test_scaled_nf,
                      which=w)

if __name__ == "__main__":
    main()