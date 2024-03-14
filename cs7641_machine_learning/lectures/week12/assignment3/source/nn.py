def gridsearch_nn(X_train,
                    y_train,
                    X_test,
                    y_test,
                  param_grid):

    nn = MLPClassifier(random_state=123)
    grid_search = GridSearchCV(nn,
                                param_grid,
                                cv=4,
                                scoring='recall',
                                n_jobs=-1).fit(X_train, y_train)                    
    cvd_nn_pred = grid_search.predict(X_test)

    grid_search_best = {'best_params': grid_search.best_params_,
                        'best_score': grid_search.best_score_,
                        'y_pred': cvd_nn_pred}

    return grid_search_best

def main():


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

    # CVD 
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')

    # Run CVD Model
    cvd_grid_nn = gridsearch_nn(X_train_scaled_cd,
                                X_test_scaled_cd,
                                y_train_cd,
                                param_grid)

    # NF
    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')

    # Run NF Model
    nf_grid_nn = gridsearch_nn(X_train_scaled_nf,
                               X_test_scaled_nf,
                               y_train_nf,
                               param_grid)

if __name__ == "__main__":
    main()