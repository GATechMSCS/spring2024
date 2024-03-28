# load, preprocess, scale, baseline
from wrangle import final_dataset

# manipulate data
import pandas as pd
import numpy as np

# visualize data
import matplotlib.pyplot as plt


import dimensionality_reduction as dr

from scipy.stats import kurtosis

from sklearn.metrics import root_mean_squared_error
import seaborn as sns

np.random.seed(123)

home_loc = f"/home/leonardo_leads/Documents/SchoolDocs/"
gt_loc = f'{home_loc}ga_tech_masters/omscs_ml/spring2024/'
course_assign = f'{gt_loc}cs7641_machine_learning/lectures/week12/assignment3/source/'

def dr_metrics(X_train, dset):

    output = {'Explained Variance': {},
                'Explained Variance Ratio': {},
                'Kurtosis': {},
                'srp_reconstruction': {},
                'Reconstruction Error': {},
                'N Neighbors': {}}

    for ncomponents in range(1, 6):
        
        # pca
        pca, _, _ = dr.pca(X_train=X_train, X_test=None, components=ncomponents)
        exp_var = np.sum(pca.explained_variance_)
        exp_var_rat = np.sum(pca.explained_variance_ratio_)

        # ica
        _, X_new_ica, _ = dr.ica(X_train=X_train, X_test=None, components=ncomponents)
        kurt = kurtosis(a=X_new_ica)[0]

        # srp
        srp, X_train_srp, _ = dr.randomized_projections(X_train=X_train, X_test=None, components=ncomponents)
        X_reconstructed = srp.inverse_transform(X_train_srp)
        reconstruction_error = root_mean_squared_error(X_train, X_reconstructed)

        # hlle
        nneighbors = int((ncomponents * (1 + (ncomponents + 1) / 2)) + 1)
        hlle, _, _ = dr.manifold_learning(X_train=X_train, X_test=None, components=ncomponents, neighbors=nneighbors)
        recon = hlle.reconstruction_error_

        output['Explained Variance'][ncomponents] = exp_var
        output['Explained Variance Ratio'][ncomponents] = exp_var_rat
        output['Kurtosis'][ncomponents] = kurt
        output['srp_reconstruction'][ncomponents] = reconstruction_error
        output['Reconstruction Error'][ncomponents] = recon
        output['N Neighbors'][ncomponents] = nneighbors
        
    return output

def optimal_ncomponents(X_train, dset):
    
    xlabel = 'Number of Components'
    title = 'Determining Number of Components'

    output = dr_metrics(X_train, dset)
    
    for i, (metric, ncomponent) in enumerate(output.items()):
        if metric == 'N Neighbors': continue
        
        x = list(ncomponent.keys())
        y = list(ncomponent.values())

        fig, ax = plt.subplots()#figsize=(15, 8)

        match metric:
            case 'Explained Variance': model = 'pca'
            
            case 'Explained Variance Ratio':
                model = 'pca'
                ax.axhline(0.9, label='90% goal', linestyle='--', color="black", linewidth=1)

            case 'Kurtosis': model = 'ica'

            case 'srp_reconstruction': model = 'srp'
            case 'srp_evr': model = 'srp'

            case 'Reconstruction Error':
                x2 = list(output['N Neighbors'].values())
                ax2 = ax.twiny()
                ax2.set(xlabel='N Neighbors', xticks=x2)
                model = 'ml'
            
        ax.plot(x, y, linewidth=2, color='red')
        ax.set(xlabel=xlabel, ylabel=metric, xticks=x, title=title)
        ax.grid()

        save_loc = f'{course_assign}plots/{dset}/dr/{model}/{metric}_opt_compon.png'
        plt.savefig(fname=f"{save_loc}")

        plt.tight_layout()
        plt.show()

def opt_component_plots(X_train,
                        pca_comp,
                        ica_comp,
                        srp_comp,
                        hlle_comp,
                        hlle_neigh,
                        dset):
    
    # pca
    _, X_new_pca, _ = dr.pca(X_train=X_train, X_test=None, components=pca_comp)

    # ica
    _, X_new_ica, _ = dr.ica(X_train=X_train, X_test=None, components=ica_comp)

    # srp
    _, X_train_srp, _ = dr.randomized_projections(X_train=X_train, X_test=None, components=srp_comp)

    # hlle
    _, X_train_hlle, _ = dr.manifold_learning(X_train=X_train, X_test=None, components=hlle_comp, neighbors=hlle_neigh)
    
    X_dr_list = [X_new_pca, X_new_ica,
                 X_train_srp, X_train_hlle]
    models = ['pca', 'ica', 'srp', 'ml']
    
    for Xdr, model in zip(X_dr_list, models):
        data = Xdr.iloc[:, 0:2]
        if model == 'ml':
            data = data.round(9)
        sns.pairplot(data=data, diag_kind='hist', corner=True)
        save_loc = f'{course_assign}plots/{dset}/dr/{model}/opt_comp_comparison.png'
        plt.savefig(fname=f"{save_loc}")
        plt.show()

# def pca_plot(list_of_colors, X_new, X_origin):
#     '''gets the dimensionality reducation and plots'''    
#     # Light is original data points, dark is projected data points
#     # shows how much "information" is discarded in this reduction of dimensionality.
    
#     plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.5)
#     plt.scatter(X_origin[:, 0], X_origin[:, 4], alpha=0.8)
#     plt.axis('equal')
#     #plt.savefig('images/information_discard.png')
#     plt.show()

#     plt.scatter(X_new[:, 0], X_new[:, 1],
#             c=list_of_colors, edgecolor='none', alpha=0.5,
#             cmap=plt.cm.get_cmap('seismic', 4))
#     plt.xlabel('Component 1')
#     plt.ylabel('Component 2')
#     plt.colorbar()
#     # save_loc = f'{course_assign}plots/{dset}/dr/{model}/info.png'
#     # plt.savefig(fname=f"{save_loc}")
#     plt.show()
    
def main():

    # CVD
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')
    # optimal_ncomponents(X_train=X_train_scaled_cd, dset='cvd')
    opt_component_plots(X_train_scaled_cd, 3, 4, 4, 4, 15, 'cvd')
    
    # NF
    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')
    # optimal_ncomponents(X_train=X_train_scaled_nf, dset='nf')
    opt_component_plots(X_train_scaled_nf, 4, 4, 4, 5, 21, 'nf')

if __name__ == "__main__":
    main()