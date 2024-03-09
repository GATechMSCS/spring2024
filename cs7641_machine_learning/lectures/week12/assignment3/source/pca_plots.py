# load, preprocess, scale, baseline
from wrangle import final_dataset

# manipulate data
import numpy as np

# visualize data
import matplotlib.pyplot as plt

# pca
from dimensionality_reduction import pca

def proportion_variance_explained(self, model):
    '''
    CALCULATE THE PROPORATION OF VARIANCE THAT IS EXPLAINED IN PRINCIPAL COMPONENTS
    '''
    total_variance = np.sum(model)
    cum_variance = np.cumsum(model)
    prop_var_expl = cum_variance/total_variance
    return prop_var_expl

def scree_plot(self, model):
    '''get a Scree Plot to find number of components'''
    # plot explained variance ratio in a scree plot
    plt.figure(1, figsize=(8, 6))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(model.explained_variance_, linewidth=2, color='red')
    plt.axis('tight')
    plt.xlim(0, 150)
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance')
    plt.savefig('images/scree_plot.png')
    plt.show()
    return plt

def variance_explained(self, prop_var_expl):
    '''better visualization of Scree Plot'''
    _, ax = plt.subplots(figsize=(8,6))
    ax.plot(prop_var_expl, color='red', linewidth=2, label='Explained Variance')
    ax.axhline(0.9, label='90% goal', linestyle='--', color="black", linewidth=1)
    ax.set_xlim(0, 150)
    ax.set_ylabel('Cumulative Prop. Of Explained Variance')
    ax.set_xlabel('Number Of Principal Components')
    ax.legend()
    plt.savefig('images/variance_explained.png')
    plt.show()
    return plt

def pca_plot(self, model, list_of_colors, X):
    '''gets the dimensionality reducation and plots'''
    X_pca = model.transform(X)
    # Light is original data points, dark is projected data points
    # shows how much "information" is discarded in this reduction of dimensionality.
    X_new = model.inverse_transform(X_pca)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
    plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
    plt.axis('equal')
    plt.savefig('images/information_discard.png')
    plt.show()

    projected = model.fit_transform(X)
    plt.scatter(projected[:, 0], projected[:, 1],
            c=list_of_colors, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('seismic', 5))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.savefig('images/the_data_for_86_components_kept')
    plt.show()
    return plt

def main():
    pass

if __name__ == "__main__":
    main()