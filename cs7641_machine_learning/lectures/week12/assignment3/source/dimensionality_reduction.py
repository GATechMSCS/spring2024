# load and preprocess
from wrangle import final_dataset

# manipulate data
import pandas as pd
import numpy as np

# machine learning models
from sklearn.decomposition import (PCA,
                                   FastICA)
from sklearn.random_projection import (SparseRandomProjection)
from sklearn.manifold import (LocallyLinearEmbedding)

np.random.seed(123)

# linear
def pca(X_train:pd.DataFrame, X_test:pd.DataFrame, kern='General'):
    """_summary_

    Args:
        X_train (pd.DataFrame): _description_
        X_test (pd.DataFrame): _description_
    """

    print('Fitting PCA')
    match kern:
        case 'General':
            print('Fitting General PCA')
            transformer = PCA(svd_solver='auto',
                        random_state=123)
            transformer.fit(X_train)
            X_train_transformed = transformer.transform(X_train)
            X_test_transformed = transformer.transform(X_test)

            # create train df
            train_back2_original = transformer.inverse_transform(X_train_transformed)
            X_train_transformed = pd.DataFrame(data=train_back2_original,
                                          columns=X_train.columns)

            # create test df
            test_back2_original = transformer.inverse_transform(X_test_transformed)
            X_test_transformed = pd.DataFrame(data=test_back2_original,
                                              columns=X_test.columns)
            print('Done with General PCA')
    print('Done with PCA')

    return transformer, X_train_transformed, X_test_transformed     

def ica(X_train:pd.DataFrame, X_test:pd.DataFrame, which='fast'):
    """_summary_

    Args:
        X_train (pd.DataFrame): _description_
        X_test (pd.DataFrame): _description_
    """

    print('\nFitting ICA')
    match which:
        case 'fast':
            print('Fitting and Transforming FastICA')
            transformer = FastICA(n_components=5,
                          random_state=123,
                          whiten='unit-variance')
            transformer.fit(X_train)
            X_train_transformed = transformer.transform(X_train)
            X_test_transformed = transformer.transform(X_test)

            # create train df
            train_back2_original = transformer.inverse_transform(X_train_transformed)
            X_train_transformed = pd.DataFrame(data=train_back2_original,
                                          columns=X_train.columns)

            # create test df
            test_back2_original = transformer.inverse_transform(X_test_transformed)
            X_test_transformed = pd.DataFrame(data=test_back2_original,
                                              columns=X_test.columns)
            print('Done with FastICA')
    print('Done with ICA')

    return transformer, X_train_transformed, X_test_transformed

def randomized_projections(X_train:pd.DataFrame, X_test:pd.DataFrame, which='Sparse'):
    """_summary_

    Args:
        X_train (pd.DataFrame): _description_
        X_test (pd.DataFrame): _description_
    """

    print('\nFitting and Transforming Randomized Projections')
    match which:
        case 'Sparse':
            print('Fitting and Transforing with Sparse Random Projection')
            transformer = SparseRandomProjection(n_components=3,
                                                 eps=0.9,
                                                 random_state=123)
            transformer.fit(X_train)
            X_train_transformed = transformer.transform(X_train)
            X_test_transformed = transformer.transform(X_test)
            
            # create train df
            train_back2_original = transformer.inverse_transform(X_train_transformed)
            X_train_transformed = pd.DataFrame(data=train_back2_original,
                                          columns=X_train.columns)

            # create test df
            test_back2_original = transformer.inverse_transform(X_test_transformed)
            X_test_transformed = pd.DataFrame(data=test_back2_original,
                                              columns=X_test.columns)
            print('Done with Sparse Random Projection')
    print('Done with Randomized Projections')

    return transformer, X_train_transformed, X_test_transformed

def manifold_learning(X_train:pd.DataFrame, X_test:pd.DataFrame, which='hem'):
    """_summary_

    Args:
        X_train (pd.DataFrame): _description_
        X_test (pd.DataFrame): _description_
    """

    print('\nFitting and Transforming Manifold Learning')
    match which:
        case 'hem':
            print('Fitting and Transforming with Locally Linear Embedding: Heissan Mapping')
            transformer = LocallyLinearEmbedding(n_neighbors=20,
                                               n_components=2,
                                               reg=1e-3,
                                               eigen_solver='dense',
                                               method='hessian',
                                               random_state=123,
                                               n_jobs=-1)
            transformer.fit(X_train)
            X_train_transformed = transformer.transform(X_train)
            X_test_transformed = transformer.transform(X_test)

            # get components and create train df
            cols = [f'Component{i}' for i in range(transformer.embedding_.shape[1])]
            X_train_transformed = pd.DataFrame(data=X_train_transformed,
                                          columns=cols)

            # create test df
            X_test_transformed = pd.DataFrame(data=X_test_transformed,
                                              columns=cols)
            print('Done with Locally Linear Embedding: Heissan Mapping')
    print('Done with Manifold Learning')

    return transformer, X_train_transformed, X_test_transformed

def main():

    # cvd
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')
    
    # nf
    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')

    # pca
    for k in ['General']:
        pca(X_train=X_train_scaled_cd, X_test=X_test_scaled_cd, kern=k)
        pca(X_train=X_train_scaled_nf, X_test=X_test_scaled_nf, kern=k)

    # ica
    for w in ['fast']:
        ica(X_train=X_train_scaled_cd, X_test=X_test_scaled_cd, which=w)
        ica(X_train=X_train_scaled_nf, X_test=X_test_scaled_nf, which=w)

    # randomized projections
    for w in ['Sparse']:
        randomized_projections(X_train=X_train_scaled_cd, X_test=X_test_scaled_cd, which=w)
        randomized_projections(X_train=X_train_scaled_nf, X_test=X_test_scaled_nf, which=w)

        # manifold learning
    for w in ['lle']:
        manifold_learning(X_train=X_train_scaled_cd,
                          X_test=X_test_scaled_cd,
                          which=w)
        manifold_learning(X_train=X_train_scaled_nf,
                          X_test=X_test_scaled_nf,
                          which=w)

if __name__ == "__main__":
    main()