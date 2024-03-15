# load and preprocess
from wrangle import final_dataset

# manipulate data
import pandas as pd
import numpy as np

# machine learning models
# clustering
from sklearn.cluster import (KMeans)

# expection maximization
from sklearn.mixture import (GaussianMixture)

# model evaluation
from sklearn.metrics import (rand_score, # lables known
                            mutual_info_score,# lables known
                            silhouette_score, # lables not known
                            calinski_harabasz_score) # lables not known

np.random.seed(123)

def expectation_maximization(X_train:pd.DataFrame, X_test:pd.DataFrame, components, which='Gaussian'):
    """_summary_

    Args:
        X_train (pd.DataFrame): _description_
        X_test (pd.DataFrame): _description_
    """

    print('Fitting and Predicting Expectation Maximization')
    match which:
        case 'Gaussian':
            print('Fitting and Predicting Gaussian Mixture')

            # creating the train object
            gm_train = GaussianMixture(n_components=components,
                                        random_state=123).fit(X_train)

            # get train labels
            X_train['mixture_clusters'] = gm_train.predict(X_train)

            # get test labels
            X_test['mixture_clusters'] = gm_train.predict(X_test)
            
            print('Done with Gaussian Mixture')
    print('Done with Expectation Maximization\n')
    return gm_train, X_train, X_test

def cluster_model(X_train:pd.DataFrame, X_test:pd.DataFrame, n_clusters, which='kmeans'):
    """_summary_

    Args:
        X_train (pd.DataFrame): _description_
        X_test (pd.DataFrame): _description_
    """
    print('Fitting and Predicting Clustering')
    match which:
        case 'kmeans':
            print('Fitting and Predicting KMeans')
            
            # creating the train object
            clustering_train = KMeans(n_clusters=n_clusters,
                                        max_iter=500,
                                        random_state=123).fit(X_train)

            # get train labels
            X_train['feat_clusters'] = clustering_train.labels_

            # get test labels
            X_test['feat_clusters'] = clustering_train.predict(X_test)
            
            print('Done with KMeans')
    print('Done with Clustering')
            
    return clustering_train, X_train, X_test

def main():

    # CVD
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')

    # NF
    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')

    # expectation maximization
    for w in ['Gaussian']:
        expectation_maximization(X_train=X_train_scaled_cd,
                            X_test=X_test_scaled_cd,
                            which=w)

        expectation_maximization(X_train=X_train_scaled_nf,
                                X_test=X_test_scaled_nf,
                                which=w)

    # clustering
    for w in ['kmeans', 'meanshift', 'ac', 'birch']:
        cluster_model(X_train=X_train_scaled_cd,
                        X_test=X_test_scaled_cd,
                        which=w)
    
        cluster_model(X_train=X_train_scaled_nf,
                        X_test=X_test_scaled_nf,
                        which=w)

if __name__ == "__main__":
    main()