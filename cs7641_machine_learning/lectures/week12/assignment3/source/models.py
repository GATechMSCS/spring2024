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
                      'step5': {}}}

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

            case 3:
                print(f'Step: {step}')
                # DR
                pca_train_copy = results[dset]['step2']['pca'][1].copy()
                ica_train_copy = results[dset]['step2']['ica'][1].copy()
                sparseRP_train_copy = results[dset]['step2']['sparseRP'][1].copy()
                manifold_train_copy = results[dset]['step2']['manifold'][1].copy()
                
                # pca
                gm_pca = cl.expectation_maximization(X_train=pca_train_copy, X_test=X_test, which='Gaussian')
                clustering_pca = cl.cluster_model(X_train=pca_train_copy, X_test=X_test, which='kmeans')

                # ica
                gm_ica = cl.expectation_maximization(X_train=ica_train_copy, X_test=X_test, which='Gaussian')
                clustering_ica = cl.cluster_model(X_train=ica_train_copy, X_test=X_test, which='kmeans')

                # srp
                gm_srp = cl.expectation_maximization(X_train=sparseRP_train_copy, X_test=X_test, which='Gaussian')
                clustering_srp = cl.cluster_model(X_train=sparseRP_train_copy, X_test=X_test, which='kmeans')

                # manifold learning
                gm_maniL = cl.expectation_maximization(X_train=manifold_train_copy, X_test=X_test, which='Gaussian')
                clustering_maniL = cl.cluster_model(X_train=manifold_train_copy, X_test=X_test, which='kmeans')

                # gm
                results[dset]['step3']['pca']['gm'] = gm_pca
                results[dset]['step3']['ica']['gm'] = gm_ica
                results[dset]['step3']['sparseRP']['gm'] = gm_srp
                results[dset]['step3']['manifold']['gm'] = gm_maniL

                # kmeans
                results[dset]['step3']['pca']['kmeans'] = clustering_pca
                results[dset]['step3']['ica']['kmeans'] = clustering_ica
                results[dset]['step3']['sparseRP']['kmeans'] = clustering_srp
                results[dset]['step3']['manifold']['kmeans'] = clustering_maniL
                print(f'Step: {step} Complete\n')

            
            case 4:
                if dset == 'nf':
                    print(f'Step: {step} ({dset} only)')
                    # DR
                    pca_train = results[dset]['step2']['pca'][1]
                    pca_test = results[dset]['step2']['pca'][2]
                    ica_train = results[dset]['step2']['ica'][1]
                    ica_test = results[dset]['step2']['ica'][2]
                    sparseRP_train = results[dset]['step2']['sparseRP'][1]
                    sparseRP_test = results[dset]['step2']['sparseRP'][2]
                    manifold_train = results[dset]['step2']['manifold'][1]
                    manifold_test = results[dset]['step2']['manifold'][2]

                    # pca
                    cvd_nn_pca = MLPClassifier(random_state=123,
                                            learning_rate_init=0.01,
                                            batch_size=65,
                                            hidden_layer_sizes=125,
                                            max_iter=200).fit(pca_train, y_train)
                    cvd_nn_pca_pred = cvd_nn_pca.predict(pca_test)

                    # ica
                    cvd_nn_ica = MLPClassifier(random_state=123,
                                            learning_rate_init=0.01,
                                            batch_size=65,
                                            hidden_layer_sizes=125,
                                            max_iter=200).fit(ica_train, y_train)
                    cvd_nn_ica_pred = cvd_nn_ica.predict(ica_test)

                    # sparse RP
                    cvd_nn_sparseRP = MLPClassifier(random_state=123,
                                            learning_rate_init=0.01,
                                            batch_size=65,
                                            hidden_layer_sizes=125,
                                            max_iter=200).fit(sparseRP_train, y_train)
                    cvd_nn_sparseRP_pred = cvd_nn_sparseRP.predict(sparseRP_test)

                    # manifold
                    cvd_nn_manifold = MLPClassifier(random_state=123,
                                            learning_rate_init=0.01,
                                            batch_size=65,
                                            hidden_layer_sizes=125,
                                            max_iter=200).fit(manifold_train, y_train)
                    cvd_nn_manifold_pred = cvd_nn_manifold.predict(manifold_test)

                    results[dset]['step4']['pca'] = cvd_nn_pca_pred
                    results[dset]['step4']['ica'] = cvd_nn_ica_pred
                    results[dset]['step4']['sparseRP'] = cvd_nn_sparseRP_pred
                    results[dset]['step4']['manifold'] = cvd_nn_manifold_pred
                    print(f'Step: {step} Complete\n')

            case 5:
                print(f'Step: {step}')
                # TODO: report on CL and NN
                cvd_model = MLPClassifier(random_state=123,
                                          learning_rate_init=0.01,
                                          batch_size=65,
                                          hidden_layer_sizes=125,
                                          max_iter=200)
                #results[dset]['step5'] = 
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