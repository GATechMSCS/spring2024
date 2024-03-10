# load and preprocess
from wrangle import final_dataset

# manipulate data
import pandas as pd
import numpy as np

# machine learning models
# clustering
from sklearn.cluster import (KMeans,
                            MeanShift,
                            AgglomerativeClustering,
                            Birch)

# expection maximization
from sklearn.mixture import (GaussianMixture)

# model evaluation
from sklearn.metrics import (rand_score, # lables known
                             mutual_info_score,# lables known
                             silhouette_score, # lables not known
                             calinski_harabasz_score) # lables not known

np.random.seed(123)

def expectation_maximization(X_train:pd.DataFrame, X_test:pd.DataFrame, which='Gaussian'):
    """_summary_

    Args:
        X_train (pd.DataFrame): _description_
        X_test (pd.DataFrame): _description_
    """

    print('Fitting and Predicting Expectation Maximization')
    match which:
        case 'Gaussian':
            print('Fitting and Predicting Gaussian Mixture')
            gm = GaussianMixture(n_components=2, random_state=0)
            gm.fit(X_train)
            gm.means_
            gm.predict(X_test)
            print('Done with Gaussian Mixture')
    print('Done with Expectation Maximization\n')
    return 1

def cluster_model(X_train:pd.DataFrame, X_test:pd.DataFrame, which='kmeans'):
    """_summary_

    Args:
        X_train (pd.DataFrame): _description_
        X_test (pd.DataFrame): _description_
    """
    print('Fitting and Predicting Clustering')
    match which:
        case 'kmeans':
            print('Fitting and Predicting KMeans')
            # creating the object
            clustering = KMeans(n_clusters=8, max_iter=500)

            # fitting the object
            clustering.fit(X_train)

            # predict on train
            y_kmeans_train = clustering.predict(X_train)
            X_train['kmeans_feat_clusters'] = y_kmeans_train

            #predict on test
            y_kmeans_test = clustering.predict(X_test)
            X_test['kmeans_feat_clusters'] = y_kmeans_test
            print('Done with KMeans')

        case 'meanshift':
            print('Fitting and Predicting MeanShift')
            
            # creating the object
            clustering = MeanShift(bandwidth=2, max_iter=500, n_jobs=-1)
            clustering.fit(X_train)

            # predict on train
            y_meanshift_train = clustering.predict(X_train)
            X_train['meanshift_feat_clusters'] = y_meanshift_train

            # predict on test
            y_meanshift_test = clustering.predict(X_test)
            X_test['meanshift_feat_clusters'] = y_meanshift_test
            print('Done with MeanShift')            

        case 'ac':
            print('Fitting and Predicting Agglomerative Clustering')
            
            # creating the object
            clustering = AgglomerativeClustering()
            y_ac_train = clustering.fit_predict(X_train)

            X_train['ac_feat_clusters'] = y_ac_train
            print('Done with Agglomerative Clustering')

        case 'birch':
            print('Fitting and Predicting Birch')
            
            # creating the object
            clustering = Birch()
            y_birch_train = clustering.fit_predict(X_train)

            X_train['birch_feat_clusters'] = y_birch_train
            print('Done with Birch')
    print('Done with Clustering')
            
    return 1

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