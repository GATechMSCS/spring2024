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

def put_it_all_together(X_train:pd.DataFrame, X_test:pd.DataFrame):

    print('\nRunning All Steps')
    
    steps = range(1, 6, 1)

    for step in steps:
            
        match step:
            case 1:
                print(f'Step: {step}')
                cl.expectation_maximization(X_train=X_train, X_test=X_test)
                cl.cluster_model(X_train=X_train, X_test=X_test)
                print(f'Step: {step} Complete\n')

            case 2:
                print(f'Step: {step}')
                dr.pca(X_train=X_train, X_test=X_test)
                dr.ica(X_train=X_train, X_test=X_test)
                dr.randomized_projections(X_train=X_train, X_test=X_test)
                dr.manifold_learning(X_train=X_train, X_test=X_test)
                print(f'Step: {step} Complete\n')

            case 3:
                print(f'Step: {step}')
                # TODO: report on CL and DR
                cl.expectation_maximization(X_train=X_train, X_test=X_test)
                cl.cluster_model(X_train=X_train, X_test=X_test)
                print(f'Step: {step} Complete\n')

            case 4:
                print(f'Step: {step}')
                # TODO: report on DR
                cvd_model = MLPClassifier(random_state=123,
                                          learning_rate_init=0.01,
                                          batch_size=65,
                                          hidden_layer_sizes=125,
                                          max_iter=200)
                print(f'Step: {step} Complete\n')

            case 5:
                print(f'Step: {step}')
                # TODO: report on CL and NN
                cvd_model = MLPClassifier(random_state=123,
                                          learning_rate_init=0.01,
                                          batch_size=65,
                                          hidden_layer_sizes=125,
                                          max_iter=200)
                print(f'Step: {step} Complete')

    print('Completed All Steps')

def main():
    results = {'cvd': {'step 1': None,
                       'step 2': None,
                       'step 3': None,
                       'step 4': None,
                       'step 5': None},
               'nf': {'step 1': None,
                      'step 2': None,
                      'step 3': None,
                      'step 4': None,
                      'step 5': None}}

    for df in results.keys():

        # CVD Then NF
        # Run All Steps
        get_data = f"X_train_scaled_{df}, X_test_scaled_{df}, y_train_{df}, y_test_{df} = final_dataset(dataset='{df}')"
        run_models = f"final_dict = put_it_all_together(X_train=X_train_scaled_{df}, X_test=X_test_scaled_{df})"
        exec(get_data)
        exec(run_models)

if __name__ == "__main__":
    main()