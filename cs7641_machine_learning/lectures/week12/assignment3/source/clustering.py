# load and preprocess
from wrangle import final_dataset

# manipulate data
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
                             davies_bouldin_score,
                             )

from sklearn.metrics.cluster import (contingency_matrix,
                                     pair_confusion_matrix)


def main():
    pass

if __name__ == "__main__":
    main()