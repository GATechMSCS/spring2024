# load and preprocess
from wrangle import final_dataset

# manipulate data
import pandas as pd
import numpy as np

# clustering
import clustering as cl

# dimentionality reduction
import dimensionality_reduction as dr

# Baseline Model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

np.random.seed(123)

def put_it_all_together(X_train:pd.DataFrame,
                        y_train:pd.Series,
                        X_test:pd.DataFrame,
                        y_test:pd.Series,
                        dset:str):

    results = {dset: {'step1': {'gm': None,
                                'kmeans': None},
                        'step2': {'pca': None,
                                'ica': None,
                                'sparseRP': None,
                                'manifold': None},
                        'step3': {'pca': {'gm': None ,
                                        'kmeans': None},
                                'ica': {'gm': None,
                                        'kmean': None},
                                'sparseRP': {'gm': None,
                                            'kmeans': None},
                                'manifold': {'gm': None,
                                            'kmeans': None}},
                        'step4': {'pca': None ,
                                'ica': None,
                                'sparseRP': None,
                                'manifold': None},
                        'step5': {'gm': None,
                                'kmean': None}}}

    param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],  # size of hidden layers
    'activation': ['relu', 'tanh', 'sigmoid'],  # activation functions
    'solver': ['adam'],  # solver for weight optimization
    'alpha': [0.0001, 0.001, 0.01],  # L2 penalty (regularization term)
    'learning_rate': ['constant', 'adaptive'],  # learning rate schedule
    'learning_rate_init': [0.001, 0.01, 0.1],  # initial learning rate
    'max_iter': [200, 500, 1000],  # maximum number of iterations
    'batch_size': [32, 64, 128],  # size of minibatches for stochastic optimizers
    'tol': [1e-3, 1e-4, 1e-5],  # tolerance for stopping criteria
    'early_stopping': [True, False],  # whether to use early stopping to prevent overfitting
    'validation_fraction': [0.1, 0.2, 0.3],  # fraction of training data to use for validation
    'n_iter_no_change': [5, 10, 20],}  # maximum number of epochs with no improvement to wait before stopping


    steps = range(1, 6, 1)

    print('\nRunning All Steps')

    for step in steps:
        match step:
            case 1:
                print(f'Step: {step}')
                
                X_train_copy1 = X_train.copy()
                X_test_copy1 = X_test.copy()
                gm = cl.expectation_maximization(X_train=X_train_copy1, X_test=X_test_copy1, which='Gaussian')
                clustering = cl.cluster_model(X_train=X_train_copy1, X_test=X_test_copy1, which='kmeans')
                results[dset]['step1']['gm'] = gm
                results[dset]['step1']['kmeans'] = clustering
                
                print(f'Step: {step} Complete\n')

            case 2:
                print(f'Step: {step}')
            
                gpca = dr.pca(X_train=X_train, X_test=X_test)
                fica = dr.ica(X_train=X_train, X_test=X_test)
                srp= dr.randomized_projections(X_train=X_train, X_test=X_test)
                lleh = dr.manifold_learning(X_train=X_train, X_test=X_test)
                results[dset]['step2']['pca'] = gpca
                results[dset]['step2']['ica'] = fica
                results[dset]['step2']['sparseRP'] = srp
                results[dset]['step2']['manifold'] = lleh
            
                print(f'Step: {step} Complete\n')

            # case 3:
            #     print(f'Step: {step}')
            
            #     # DR
            #     pca_train_copy = results[dset]['step2']['pca'][1].copy()
            #     ica_train_copy = results[dset]['step2']['ica'][1].copy()
            #     sparseRP_train_copy = results[dset]['step2']['sparseRP'][1].copy()
            #     manifold_train_copy = results[dset]['step2']['manifold'][1].copy()
                
            #     # pca
            #     gm_pca = cl.expectation_maximization(X_train=pca_train_copy, X_test=X_test, which='Gaussian')
            #     clustering_pca = cl.cluster_model(X_train=pca_train_copy, X_test=X_test, which='kmeans')

            #     # ica
            #     gm_ica = cl.expectation_maximization(X_train=ica_train_copy, X_test=X_test, which='Gaussian')
            #     clustering_ica = cl.cluster_model(X_train=ica_train_copy, X_test=X_test, which='kmeans')

            #     # srp
            #     gm_srp = cl.expectation_maximization(X_train=sparseRP_train_copy, X_test=X_test, which='Gaussian')
            #     clustering_srp = cl.cluster_model(X_train=sparseRP_train_copy, X_test=X_test, which='kmeans')

            #     # manifold learning
            #     gm_maniL = cl.expectation_maximization(X_train=manifold_train_copy, X_test=X_test, which='Gaussian')
            #     clustering_maniL = cl.cluster_model(X_train=manifold_train_copy, X_test=X_test, which='kmeans')

            #     # gm
            #     results[dset]['step3']['pca']['gm'] = gm_pca
            #     results[dset]['step3']['ica']['gm'] = gm_ica
            #     results[dset]['step3']['sparseRP']['gm'] = gm_srp
            #     results[dset]['step3']['manifold']['gm'] = gm_maniL

            #     # kmeans
            #     results[dset]['step3']['pca']['kmeans'] = clustering_pca
            #     results[dset]['step3']['ica']['kmeans'] = clustering_ica
            #     results[dset]['step3']['sparseRP']['kmeans'] = clustering_srp
            #     results[dset]['step3']['manifold']['kmeans'] = clustering_maniL
            
            #     print(f'Step: {step} Complete\n')

            
            case 4:
                if dset == 'cvd':
                    print(f'Step: {step} ({dset} only)')
            
                    pca_train = results[dset]['step2']['pca'][1]
                    pca_test = results[dset]['step2']['pca'][2]
                    ica_train = results[dset]['step2']['ica'][1]
                    ica_test = results[dset]['step2']['ica'][2]
                    sparseRP_train = results[dset]['step2']['sparseRP'][1]
                    sparseRP_test = results[dset]['step2']['sparseRP'][2]
                    manifold_train = results[dset]['step2']['manifold'][1]
                    manifold_test = results[dset]['step2']['manifold'][2]

                    print('\nFitting and Predicting PCA NN')
                    cvd_nn_pca = MLPClassifier(random_state=123)    
                    grid_search_pca = GridSearchCV(cvd_nn_pca,
                                                param_grid,
                                                cv=4,
                                                scoring='recall',
                                                n_jobs=-1).fit(pca_train, y_train)
                    cvd_nn_pca_pred = grid_search_pca.predict(pca_test)
                    print('PCA NN Complete')

                    print('\nFitting and Predicting ICA NN')
                    cvd_nn_ica = MLPClassifier(random_state=123)    
                    grid_search_ica = GridSearchCV(cvd_nn_ica,
                                                param_grid,
                                                cv=4,
                                                scoring='recall',
                                                n_jobs=-1).fit(ica_train, y_train)
                    cvd_nn_ica_pred = grid_search_ica.predict(ica_test)
                    print('ICA NN Complete')

                    print('\nFitting and Predicting Sparse RP NN')
                    cvd_nn_sparseRP = MLPClassifier(random_state=123)    
                    grid_search_RP = GridSearchCV(cvd_nn_sparseRP,
                                                param_grid,
                                                cv=4,
                                                scoring='recall',
                                                n_jobs=-1).fit(sparseRP_train, y_train)                    
                    cvd_nn_sparseRP_pred = grid_search_RP.predict(sparseRP_test)
                    print('Sparse RP NN Complete')

                    print('\nFitting and Predicting Maniforld Learning NN')
                    cvd_nn_manifold = MLPClassifier(random_state=123)    
                    grid_search_manifold = GridSearchCV(cvd_nn_manifold,
                                                param_grid,
                                                cv=4,
                                                scoring='recall',
                                                n_jobs=-1).fit(manifold_train, y_train)                    
                    cvd_nn_manifold_pred = grid_search_manifold.predict(manifold_test)
                    print('Maniforld Learning NN Complete')

                    grid_search_pca_best = {'best_params': grid_search_pca.best_params_, 
                                            'best_score': grid_search_pca.best_score_,
                                            'y_pred': cvd_nn_pca_pred}
                    grid_search_ica_best = {'best_params': grid_search_ica.best_params_, 
                                            'best_score': grid_search_ica.best_score_,
                                            'y_pred': cvd_nn_ica_pred}
                    grid_search_RP_best = {'best_params': grid_search_RP.best_params_, 
                                            'best_score': grid_search_RP.best_score_,
                                            'y_pred': cvd_nn_sparseRP_pred}
                    grid_search_manifold_best = {'best_params': grid_search_manifold.best_params_, 
                                                'best_score': grid_search_manifold.best_score_,
                                                'y_pred': cvd_nn_manifold_pred}

                    results[dset]['step4']['pca'] = grid_search_pca_best
                    results[dset]['step4']['ica'] = grid_search_ica_best
                    results[dset]['step4']['sparseRP'] = grid_search_RP_best
                    results[dset]['step4']['manifold'] = grid_search_manifold_best
            
                    print(f'Step: {step} Complete\n')

            case 5:
                if dset == 'cvd':
                    print(f'Step: {step} ({dset}) only')

                    X_train_gm = results[dset]['step1']['gm'][1]
                    X_train_cl = results[dset]['step1']['kmeans'][1]

                    X_test_gm = results[dset]['step1']['gm'][3]
                    X_test_cl = results[dset]['step1']['kmeans'][3]

                    print('Fitting and Predicting GM NN')
                    cvd_nn_gm = MLPClassifier(random_state=123)
                    grid_search_gm = GridSearchCV(cvd_nn_gm,
                                                param_grid,
                                                cv=4,
                                                scoring='recall',
                                                n_jobs=-1).fit(X_train_gm, y_train)                    
                    cvd_nn_gm_pred = grid_search_gm.predict(X_test_gm)
                    print('GM NN Complete')

                    print('Fitting and Predicting Clustering NN')
                    cvd_nn_cl = MLPClassifier(random_state=123)
                    grid_search_cl = GridSearchCV(cvd_nn_cl,
                                                param_grid,
                                                cv=4,
                                                scoring='recall',
                                                n_jobs=-1).fit(X_train_cl, y_train)                    
                    cvd_nn_cl_pred = grid_search_gm.predict(X_test_cl)
                    print('Clustering NN Complete')

                    grid_search_gm_best = {'best_params': grid_search_gm.best_params_, 
                                            'best_score': grid_search_gm.best_score_,
                                            'y_pred': cvd_nn_gm_pred}
                    grid_search_cl_best = {'best_params': grid_search_cl.best_params_, 
                                                'best_score': grid_search_cl.best_score_,
                                                'y_pred': cvd_nn_cl_pred}
                    
                    results[dset]['step5']['gm'] = grid_search_gm_best
                    results[dset]['step5']['kmeans'] = grid_search_cl_best
                    
                    print(f'Step: {step} Complete\n')

    print('Completed All Steps')

    return results

def main():

    # CVD 
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')

    # Run CVD Model
    results_cv = put_it_all_together(X_train=X_train_scaled_cd,
                                    y_train=y_train_cd,
                                    X_test=X_test_scaled_cd,
                                    y_test=y_test_cd,
                                    dset='cvd')

    # NF
    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')

    # Run NF Model
    results_nf = put_it_all_together(X_train=X_train_scaled_nf,
                                    y_train=y_train_nf,
                                    X_test=X_test_scaled_nf, 
                                    y_test=y_test_nf,
                                    dset='nf')

    print(results_cv)
    print()
    print(results_nf)

if __name__ == "__main__":
    main()