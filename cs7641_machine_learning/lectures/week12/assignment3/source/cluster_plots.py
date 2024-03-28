# load, preprocess, scale, baseline
from wrangle import final_dataset

# data manipulation
import pandas as pd
import numpy as np

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# model
from sklearn.cluster import KMeans

# expection maximization
from sklearn.mixture import (GaussianMixture)

import clustering as cl

# model evaluation
from sklearn.metrics import (adjusted_rand_score, # lables known
                            adjusted_mutual_info_score,# lables known
                            silhouette_score, # lables not known
                            calinski_harabasz_score) # lables not known

# manipulate data
import pandas as pd
import numpy as np

np.random.seed(123)

home_loc = f"/home/leonardo_leads/Documents/SchoolDocs/"
gt_loc = f'{home_loc}ga_tech_masters/omscs_ml/spring2024/'
course_assign = f'{gt_loc}cs7641_machine_learning/lectures/week12/assignment3/source/'

# GAUSSIAN MIXTURE
def gm_metrics(X_train, y_train):
    output = {'AIC': {},
            'BIC': {},
            'Silhouette Score': {},
            'Calinski Score': {},
            'Adjusted Rand Score': {},
            'Adjusted Mutual Info Score': {}}

    for components in range(2, 14):
        gm_train, gm_mixture = cl.expectation_maximization(X_train=X_train, X_test=None, 
                                         components=components)

        output['AIC'][components] = gm_train.aic(X_train)
        output['BIC'][components] = gm_train.bic(X_train)
        output['Silhouette Score'][components] = silhouette_score(X_train, gm_mixture)
        output['Calinski Score'][components] = calinski_harabasz_score(X_train, gm_mixture)
        output['Adjusted Rand Score'][components] = adjusted_rand_score(y_train, gm_mixture)
        output['Adjusted Mutual Info Score'][components] = adjusted_mutual_info_score(y_train, gm_mixture)

    return output

def ncomponents_optimal(X_train, y_train, dset, model, step3):

    output = gm_metrics(X_train, y_train)
    
    xlabel = 'Number of Components'
    title = 'Determining Number of Components'
    
    for i, (metric, ncomponent_score) in enumerate(output.items()):

        fig, ax = plt.subplots()#figsize=(15, 8)

        x = list(ncomponent_score.keys())
        y = list(ncomponent_score.values())

        ax.plot(x, y)
        ax.set(xlabel=xlabel, ylabel=metric, xticks=range(1, 14), title=title)
        ax.grid()

        
        if step3:   
            save_loc = f'{course_assign}plots/{dset}/clustering/step3/gaussian_mixture/{model}/{model}_{metric}.png'
        else:
            save_loc = f'{course_assign}plots/{dset}/clustering/gaussian_mixture/{model}_{metric}.png'        
        plt.savefig(fname=f"{save_loc}")

        plt.tight_layout()
        plt.show()

def scatter_component_means(X_train, components, dset, xlabel, ylabel, model, step3):

    gm_train, gm_mixture = cl.expectation_maximization(X_train=X_train, X_test=None, 
                                                    components=components)
    means = gm_train.means_

    x = X_train.iloc[:, 0]
    y = X_train.iloc[:, 1]

    plt.scatter(x, y, c=gm_mixture, s=20, cmap='viridis')
    plt.scatter(means[:, 0], means[:, 1], c='red', s=500, alpha=1, marker='x')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(label='Component Centers for Two Columns')

    if step3:
        save_loc = f'{course_assign}plots/{dset}/clustering/step3/gaussian_mixture/{model}/{model}_component_centers.png'
    else:
        save_loc = f'{course_assign}plots/{dset}/clustering/gaussian_mixture/{model}_component_centers.png'
    plt.savefig(fname=f"{save_loc}")

    plt.tight_layout()
    plt.show()

##### KMEANS
def cluster_metrics(X_train, y_train):
    output = {'Inertia': {},
            'Silhouette Score': {},
            'Calinski Score': {},
            'Adjusted Rand Score': {},
            'Adjusted Mutual Info Score': {}}

    for nclusters in range(2, 14):
        clustering_train, km_cluster = cl.cluster_model(X_train=X_train, X_test=None, n_clusters=nclusters)
        
        output['Inertia'][nclusters] = clustering_train.inertia_
        output['Silhouette Score'][nclusters] = silhouette_score(X_train, km_cluster)
        output['Calinski Score'][nclusters] = calinski_harabasz_score(X_train, km_cluster)
        output['Adjusted Rand Score'][nclusters] = adjusted_rand_score(y_train, km_cluster)
        output['Adjusted Mutual Info Score'][nclusters] = adjusted_mutual_info_score(y_train, km_cluster)

    return output

def nclusters_optimal(X_train, y_train, dset, model, step3):

    output = cluster_metrics(X_train, y_train)
    
    xlabel = 'Number of Clusters'
    title = 'Determining Number of Clusters'
    
    for i, (metric, nclust_score) in enumerate(output.items()):

        fig, ax = plt.subplots()#figsize=(15, 8)

        x = list(nclust_score.keys())
        y = list(nclust_score.values())

        ax.plot(x, y)
        ax.set(xlabel=xlabel, ylabel=metric, xticks=range(1, 14), title=title)
        ax.grid()
        if step3:
            save_loc = f'{course_assign}plots/{dset}/clustering/step3/kmeans/{model}/{model}_{metric}.png'
        else:
            save_loc = f'{course_assign}plots/{dset}/clustering/kmeans/{model}_{metric}.png'
        plt.savefig(fname=f"{save_loc}")

        plt.tight_layout()
        plt.show()

def scatter_cluster_centers(X_train, k, dset, xlabel, ylabel, model, step3):

    clustering_train, km_cluster = cl.cluster_model(X_train=X_train, X_test=None, n_clusters=k)
    centers = clustering_train.cluster_centers_

    x = X_train.iloc[:, 0]
    y = X_train.iloc[:, 1]

    plt.scatter(x, y, c=km_cluster, s=20, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=500, alpha=1, marker='x')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(label='Cluster Centers for Two Columns')

    if step3:
        save_loc = f'{course_assign}plots/{dset}/clustering/step3/kmeans/{model}/{model}_cluster_centers.png'
    else:
        save_loc = f'{course_assign}plots/{dset}/clustering/kmeans/{model}_cluster_centers.png'
    plt.savefig(fname=f"{save_loc}")

    plt.tight_layout()
    plt.show()

def main():

    # CVD 
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')

    # GMM
    ncomponents_optimal(X_train_scaled_cd, y_train_cd, 'cvd')
    #scatter_component_means(X_train_scaled_cd, 5, 'cvd', 'Height', 'Weight')

    # KMEANS
    nclusters_optimal(X_train_scaled_cd, y_train_cd, 'cvd')
    #scatter_cluster_centers(X_train_scaled_cd, 3,'cvd')
    
    # NF
    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')

    # GMM
    ncomponents_optimal(X_train_scaled_nf, y_train_nf, 'nf')
    #scatter_component_means(X_train_scaled_nf, 5, 'nf', 'Protein', 'Carbohydrates')

    # KMEANS
    nclusters_optimal(X_train_scaled_nf, y_train_nf, 'nf')
    #scatter_cluster_centers(X_train_scaled_nf, 4,'nf')

if __name__ == "__main__":
    main()