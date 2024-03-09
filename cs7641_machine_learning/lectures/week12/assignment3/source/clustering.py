# load and preprocess
from wrangle import final_dataset

# manipulate data
import pandas as pd
import numpy as np

# machine learning models
# clustering
from sklearn.cluster import (k_means,
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
from sklearn.mixture import GaussianMixture

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

def expectation_maximization(df:pd.DataFrame):
    pass

def cluster_model(df:pd.DataFrame):
    pass

def main():

    # CVD
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')
    
    expectation_maximization(df=X_train_scaled_cd)
    cluster_model(df=X_train_scaled_cd)

    # NF
    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')

    expectation_maximization(df=X_train_scaled_nf)
    cluster_model(df=X_train_scaled_cd)

if __name__ == "__main__":
    main()