# load, preprocess, scale, baseline
from wrangle import final_dataset

# manipulate data
import pandas as pd
import numpy as np

# visualize data
import matplotlib.pyplot as plt

from sklearn.decomposition import (PCA,
                                    FastICA)
from sklearn.random_projection import (SparseRandomProjection)
from sklearn.manifold import (LocallyLinearEmbedding)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from scipy.stats import kurtosis

from sklearn.metrics import recall_score, balanced_accuracy_score

np.random.seed(123)

home_loc = f"/home/leonardo_leads/Documents/SchoolDocs/"
gt_loc = f'{home_loc}ga_tech_masters/omscs_ml/spring2024/'
course_assign = f'{gt_loc}cs7641_machine_learning/lectures/week12/assignment3/source/'

def dr_metrics(X_train, y_train, X_test, y_test, dset):

    score_key = 'Recall' if dset == 'cvd' else 'Balanced Accuracy'
    output = {'Explained Variance': {},
                'Explained Variance Ratio': {},
                'Kurtosis': {},
                score_key: {} ,
                'Reconstruction Error': {},
                'N Neighbors': {}}

    for ncomponents in range(1, 6):
        
        # pca
        pca = PCA(n_components=ncomponents,
                            random_state=123).fit(X_train)
        X_new_pca = pca.transform(X_train)
        X_orgin_pca = pca.inverse_transform(X_new_pca)
        exp_var = np.sum(pca.explained_variance_)
        exp_var_rat = np.sum(pca.explained_variance_ratio_)

        # ica
        X_new_ica = FastICA(n_components=ncomponents,
                        random_state=123).fit_transform(X_train)
        kurt = kurtosis(a=X_new_ica)[0]

        # sro
        srp = SparseRandomProjection(n_components=ncomponents,
                                        random_state=123)
        srp.fit(X_train)
        X_train_srp = srp.transform(X_train)
        X_test_srp = srp.transform(X_test)
        mlp = MLPClassifier(random_state=123)
        mlp.fit(X_train_srp, y_train)
        y_pred = mlp.predict(X_test_srp)
        score = recall_score(y_test, y_pred) if dset == 'cvd' else balanced_accuracy_score(y_test, y_pred)

        # hlle
        nneighbors = int((ncomponents * (1 + (ncomponents + 1) / 2)) + 1)
        lle = LocallyLinearEmbedding(n_components=ncomponents,
                                    n_neighbors=nneighbors,
                                    method='hessian',
                                    reg=1e-3,
                                    n_jobs=-1,
                                    eigen_solver='dense',
                                    random_state=123).fit(X_train)
        recon = lle.reconstruction_error_

        output['Explained Variance'][ncomponents] = exp_var
        output['Explained Variance Ratio'][ncomponents] = exp_var_rat
        output['Kurtosis'][ncomponents] = kurt
        output[score_key][ncomponents] = score 
        output['Reconstruction Error'][ncomponents] = recon
        output['N Neighbors'][ncomponents] = nneighbors
        
    return output, X_new_pca, X_orgin_pca

def optimal_ncomponents(output):
    
    xlabel = 'Number of Components'
    title = 'Determining Number of Components'
    
    for i, (metric, ncomponent_score) in enumerate(output.items()):

        x = list(ncomponent_score.keys())
        y = list(ncomponent_score.values())

        fig, ax = plt.subplots(figsize=(20, 8))

        match metric:
            case 'Explained Variance': model = 'pca'
            
            case 'Explained Variance Ratio':
                model = 'pca'
                ax.axhline(0.9, label='90% goal', linestyle='--', color="black", linewidth=1)

            case 'Kurtosis': model = 'ica'

            case 'Recall': model = 'srp'

            case 'Balanceed Accuracy': model = 'srp'

            case 'Reconstruction Error':
                x2 = list(output['N Neighbors'].values())
                ax2 = ax.twiny()
                ax2.set_xticks(x2)
                ax2.set_xlabel('N Neighbors')
                model = 'ml'

            case 'N Neighbors':
                continue
            
        ax.plot(x, y, linewidth=2, color='red')
        ax.set(xlabel=xlabel, ylabel=metric, xticks=range(1, 6), title=title)
        ax.grid()

        save_loc = f'{course_assign}plots/{dset}/dr/{model}/optimal_components.png'
        plt.savefig(fname=f"{save_loc}")

        plt.tight_layout()
        plt.show()

def pca_plot(list_of_colors, X_new, X_origin):
    '''gets the dimensionality reducation and plots'''    
    # Light is original data points, dark is projected data points
    # shows how much "information" is discarded in this reduction of dimensionality.
    
    plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.5)
    plt.scatter(X_origin[:, 0], X_origin[:, 4], alpha=0.8)
    plt.axis('equal')
    #plt.savefig('images/information_discard.png')
    plt.show()

    plt.scatter(X_new[:, 0], X_new[:, 1],
            c=list_of_colors, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('seismic', 4))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    # save_loc = f'{course_assign}plots/{dset}/dr/{model}/info.png'
    # plt.savefig(fname=f"{save_loc}")
    plt.show()

def get_pca_plots(X_train, y_train, X_test, y_test, dset):

    ## calculations for models
    output, X_new_pca, X_orgin_pca = dr_metrics(X_train,
                                                y_train, 
                                                X_test,
                                                y_test, 
                                                dset)
    print(output)
    list_of_colors = list(range(8696))

    # plots
    optimal_ncomponents(output)
    #pca_plot(list_of_colors, X_new_pca, X_orgin_pca)
    
def main():

    # CVD
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')
    get_pca_plots(X_train_scaled_cd, y_train_cd, X_test_scaled_cd, y_test_cd, 'cvd')

    # NF
    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')
    get_pca_plots(X_train_scaled_nf, y_train_nf, X_test_scaled_nf, y_test_nf, 'nf')

if __name__ == "__main__":
    main()