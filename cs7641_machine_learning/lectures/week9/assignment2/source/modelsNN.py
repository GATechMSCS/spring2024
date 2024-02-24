# load and preprocess
from wrangle import final_dataset

# manipulate data
import numpy as np

# nn algo grid
from nn_algo_grid import nn_algo_tune

# machine learning models
import mlrose_hiive as rh

# model evaluation
from sklearn.metrics import recall_score

# timing
from time import time

np.random.seed(123)

def get_nn_algo(algo, hyper1=None, hyper2=None, hyper3=None):
    """Instantiate NN Algos

    Args:
        algo (str): Which algo to use with NN

    Returns:
        NN Object: NN object for algo
    """

    match algo:
        case 'Random Hill Climbing':
            print(f"Get_NN_Algo: {algo}")
            
            nn = rh.NeuralNetwork(hidden_nodes=[125],
                                  activation='relu',
                                  algorithm='random_hill_climb',
                                  clip_max=1e+10,
                                  restarts=hyper1,
                                  max_iters=150,
                                  random_state=123,
                                  curve=True,
                                  learning_rate=hyper2,)
            
        case 'Simulated Annealing':
            print(f"Get_NN_Algo: {algo}")
            
            nn = rh.NeuralNetwork(hidden_nodes=[125],
                                  activation='relu',
                                  algorithm='simulated_annealing',
                                  clip_max=1e+10,
                                  schedule=hyper1, 
                                  max_iters=150,
                                  random_state=123,
                                  curve = True,
                                  learning_rate=hyper2,)
            
        case 'Genetic Algorithm':
            print(f"Get_NN_Algo: {algo}")
            
            nn = rh.NeuralNetwork(hidden_nodes=[125], 
                                  activation='relu',
                                  algorithm='genetic_alg', 
                                  clip_max=1e+10, 
                                  pop_size=hyper1,
                                  mutation_prob=hyper3, 
                                  max_iters=150, 
                                  random_state=123, 
                                  curve=True,
                                  learning_rate=hyper2,)

        case 'Gradient Descent':
            print(f"Get_NN_Algo: {algo}")
            
            nn = rh.NeuralNetwork(hidden_nodes=[125], 
                                  activation='relu',
                                  algorithm='gradient_descent', 
                                  clip_max=1e+10, 
                                  max_iters=150, 
                                  random_state=123, 
                                  curve=True,
                                  learning_rate=0.01,)
            
    return nn

def good_weights(X_train, y_train, X_test, y_test):
    """Uses selected algorithm to search for optimal NN weights

    Args:
        data (pd.DataFrame): data to use for optimization
    """

    for algo, combos in nn_algo_tune.items():        
        for combo, hypers in combos.items():
            hypers_keys = list(hypers)

            print(f'Running {algo}')
            t0 = time()

            match algo:
                case 'Random Hill Climbing':
                    rs = hypers[hypers_keys[0]]
                    lr = hypers[hypers_keys[1]]
                    nn_algo_tune[algo][combo]['model'] = get_nn_algo(algo=algo,
                                                                     hyper1=rs,
                                                                     hyper2=lr)

                case 'Simulated Annealing':
                    dule = hypers[hypers_keys[0]]
                    lr = hypers[hypers_keys[1]]
                    nn_algo_tune[algo][combo]['model'] = get_nn_algo(algo=algo,
                                                                     hyper1=dule,
                                                                     hyper2=lr,)

                case 'Genetic Algorithm':
                    ps = hypers[hypers_keys[0]]
                    lr = hypers[hypers_keys[1]]
                    mp = hypers[hypers_keys[2]]
                    nn_algo_tune[algo][combo]['model'] = get_nn_algo(algo=algo,
                                                                     hyper1=ps,
                                                                     hyper2=lr,
                                                                     hyper3=mp)

                case 'Gradient Descent':
                    nn_algo_tune[algo][combo]['model'] = get_nn_algo(algo='Gradient Descent',
                                                                     hyper1=None,
                                                                     hyper2=None,
                                                                     hyper3=None)
            t1 = time()
            seconds = t1 - t0
            print(f'Instantiated {algo}\nTime (Seconds): {seconds}')

            t2 = time()
            
            print(f'\nWorking on: {algo}\nFitting: {algo}')
            t0 = time()        
            nn_algo_tune[algo][combo]['model'].fit(X_train, y_train)
            t1 = time()
            seconds1 = t1 - t0
            print(f'Model Fitting Complete. Time: {seconds1} seconds')

            # TRAINING
            print(f'\nPredicting (TRAINING): {algo}')
            y_train_pred = nn_algo_tune[algo][combo]['model'].predict(X_train)

            print(f'\nCalculating Recall (TRAINING): {algo}')
            recall_train = recall_score(y_train, y_train_pred)

            print(f'\nAppending Results (TRAINING): {algo}')
            nn_algo_tune[algo][combo]['y_train_pred'] = y_train_pred
            nn_algo_tune[algo][combo]['recall_train'] = recall_train
            
            print(f"\nRecall Score (TRAINING): {recall_train}%")

            # TESTING
            print(f'\nPredicting (TESTING): {algo}')
            y_test_pred = nn_algo_tune[algo][combo]['model'].predict(X_test)

            print(f'\nCalculting Recall (TESTING): {algo}')
            recall_test = recall_score(y_test, y_test_pred)

            print(f'\nAppending Results (TESTING): {algo}')
            nn_algo_tune[algo][combo]['y_test_pred'] = y_test_pred
            nn_algo_tune[algo][combo]['recall_test'] = recall_test
            
            print(f"\nRecall Score (TESTING): {recall_test}%\n")

    t3 = time()
    loopS = t3 - t2
    loopM = loopS / 60
    print(f'Completed Fitting and Predicting\nTime (Seconds): {loopS}\nTime (Minutes): {loopM}')

    return nn_algo_tune

def main():

    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')

    nn = good_weights(X_train_scaled_cd, y_train_cd,
                      X_test_scaled_cd, y_test_cd)

if __name__ == "__main__":
    main()