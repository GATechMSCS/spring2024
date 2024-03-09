# data manipulation
import pandas as pd
import numpy as np

# plotting
import matplotlib.pyplot as plt

# model
from sklearn.cluster import KMeans

def get_k_val_from_elbow(df):
    '''Takes in a dataframe and a list of features to cluster on and returns a plot of K value against the inertia'''
    output = {}

    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)
        output[k] = kmeans.inertia_

    ax = pd.Series(output).plot(figsize=(13, 7))
    ax.set(xlabel='k', ylabel='inertia', xticks=range(1, 20), title='The elbow method for determining k')
    ax.grid()

def plot_cluster():
    pass
    plt.scatter(clusters_train1.iloc[:, 0], clusters_train1.iloc[:, 1], c=y_kmeans1, s=20, cmap='viridis')
    centers = kmeans1.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=500, alpha=1, marker = 'x')

    # create plot of clusters and means
    sns.barplot(x=X_train["feat_clusters"], y=y_train)

def main():
    pass

if __name__ == "__main__":
    main()