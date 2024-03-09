# load and preprocess
from wrangle import final_dataset

# manipulate data
import pandas as pd
import numpy as np

# machine learning models
# clustering
from sklearn.cluster import (KMeans,
                            AffinityPropagation,
                            MeanShift,
                            SpectralClustering,
                            AgglomerativeClustering,
                            FeatureAgglomeration,
                            DBSCAN,
                            HDBSCAN,
                            OPTICS,
                            Birch)

# expection maximization
from sklearn.mixture import (GaussianMixture,
                             BayesianGaussianMixture)

# model evaluation
from sklearn.metrics import (rand_score,
                             adjusted_rand_score,
                             mutual_info_score,
                             adjusted_mutual_info_score,
                             normalized_mutual_info_score,
                             homogeneity_score,
                             completeness_score,
                             v_measure_score,
                             homogeneity_completeness_v_measure,
                             fowlkes_mallows_score,
                             silhouette_score,
                             calinski_harabasz_score,
                             davies_bouldin_score)

from sklearn.metrics.cluster import (contingency_matrix,
                                     pair_confusion_matrix)

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

        case 'Bayesian':
            print('Fitting and Predicting Bayesian Gaussian Mixture')
            bgm = BayesianGaussianMixture(n_components=2,
                                          random_state=123)
            bgm.fit(X_train)
            bgm.means_
            bgm.predict(X_train)
            bgm.predict(X_test)
            print('Done with Bayesian Gaussian Mixture')
    print('Done with Expectation Maximization\n')
    return 1

def cluster_model(X_train:pd.DataFrame, X_test:pd.DataFrame, which='kmeans'):
    """_summary_

    Args:
        X_train (pd.DataFrame): _description_
        X_test (pd.DataFrame): _description_
    """
    print('Fitting Clustering')
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
    print('Done with Clustering\n')
            
    return X_train, X_test

def main():

    # CVD
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')

    # NF
    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')

    # expectation maximization
    for w in ['Gaussian', 'Bayesian']:
        expectation_maximization(X_train=X_train_scaled_cd,
                            X_test=X_test_scaled_cd,
                            which=w)

        expectation_maximization(X_train=X_train_scaled_nf,
                             X_test=X_test_scaled_nf,
                             which=w)

    # clustering
    for w in ['kmeans', 'meanshift']:
        cluster_model(X_train=X_train_scaled_cd,
                  X_test=X_test_scaled_cd,
                  which=w)
    
        cluster_model(X_train=X_train_scaled_nf,
                  X_test=X_test_scaled_nf,
                  which=w)

if __name__ == "__main__":
    main()