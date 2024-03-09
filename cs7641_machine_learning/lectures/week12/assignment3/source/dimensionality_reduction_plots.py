# load, preprocess, scale, baseline
from wrangle import final_dataset

# manipulate data
import pandas as pd
import numpy as np

# visualize data
import matplotlib.pyplot as plt

# pca
from dimensionality_reduction import pca

np.random.seed(123)

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

def get_pca_plots(X_train:pd.DataFrame, X_test:pd.DataFrame, fit_pca):

    ## calculations for models
    explained_variance = fit_pca.explained_variance_
    list_of_colors = list(range(138))

    prop_var_expl = proportion_variance_explained(explained_variance)
    screech, var_exp, plot_pca = (scree_plot(fit_pca), 
                                  variance_explained(prop_var_expl),
                                  pca_plot(fit_pca, list_of_colors, X))

    return screech, var_exp, plot_pca
    
def main():

    # CVD
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')
    pca(X_train=X_train_scaled_cd, X_test=X_test_scaled_cd)
    get_pca_plots(X_train=X_train_scaled_cd , X_test=X_test_scaled_cd)

    # NF
    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')
    pca(X_train=X_train_scaled_nf, X_test=X_test_scaled_nf)
    get_pca_plots(X_train= X_train_scaled_nf, X_test=X_test_scaled_nf)

if __name__ == "__main__":
    main()