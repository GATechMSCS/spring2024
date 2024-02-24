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

def get_nn_algo(algo):
    """Instantiate NN Algos

    Args:
        algo (str): Which algo to use with NN

    Returns:
        NN Object: NN object for algo
    """

    match algo:
        case 'Random Hill Climbing':
            print(f"Running {algo}")
            
            nn = rh.NeuralNetwork(hidden_nodes=[125],
                                  activation='relu',
                                  algorithm='random_hill_climb',
                                  clip_max=1e+10,
                                  restarts=200,
                                  max_iters=150,
                                  random_state=123,
                                  curve=True,
                                  learning_rate=0.01,)
            
        case 'Simulated Annealing':
            print(f"Running {algo}")
            
            nn = rh.NeuralNetwork(hidden_nodes=[125],
                                  activation='relu',
                                  algorithm='simulated_annealing',
                                  clip_max=1e+10,
                                  schedule=rh.GeomDecay(), 
                                  max_iters=150,
                                  random_state=123,
                                  curve = True,
                                  learning_rate=0.01,)
            
        case 'Genetic Algorithm':
            print(f"Running {algo}")
            
            nn = rh.NeuralNetwork(hidden_nodes=[125], 
                                  activation='relu',
                                  algorithm='genetic_alg', 
                                  clip_max=1e+10, 
                                  pop_size=200,
                                  mutation_prob=0.1, 
                                  max_iters=150, 
                                  random_state=123, 
                                  curve=True,
                                  learning_rate=0.01,)

        case 'Gradient Descent':
            print(f"Running {algo}")
            
            nn = rh.NeuralNetwork(hidden_nodes=[125], 
                                  activation='relu',
                                  algorithm='gradient_descent', 
                                  clip_max=1e+10, 
                                  pop_size=200, 
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

    algos = np.array(['Random Hill Climbing',
                      'Simulated Annealing',
                      'Genetic Algorithm',
                      'Gradient Descent'])

    print('Running All Algorithms')
    t0 = time()
    nn = {algo: {'nn': get_nn_algo(algo=algo)} for algo in algos}
    t1 = time()
    seconds = t1 - t0
    minutes = seconds / 60
    print(f'Compmleted All Algorithms\nTime (Seconds): {seconds}\nTime (Minutes): {minutes} ')

    t2 = time()
    for algo in algos:
        print(f'\nWorking on: {algo}\nFitting: {algo}')
        t0 = time()        
        nn[algo]['nn'].fit(X_train, y_train)
        t1 = time()
        seconds1 = t1 - t0
        print(f'Model Fitting Complete. Time: {seconds1} seconds')

        # TRAINING
        print(f'\nPredicting (TRAINING): {algo}')
        y_train_pred = nn[algo]['nn'].predict(X_train)

        print(f'\nCalculating Recall (TRAINING): {algo}')
        recall_train = recall_score(y_train, y_train_pred)

        print(f'\nAppending Results (TRAINING): {algo}')
        nn[algo]['y_train_pred'] = y_train_pred
        nn[algo]['recall_train'] = recall_train
        
        print(f"\nRecall Score (TRAINING): {recall_train}%")

        # TESTING
        print(f'\nPredicting (TESTING): {algo}')
        y_test_pred = nn[algo]['nn'].predict(X_test)

        print(f'\nCalculting Recall (TESTING): {algo}')
        recall_test = recall_score(y_test, y_test_pred)

        print(f'\nAppending Results (TESTING): {algo}')
        nn[algo]['y_test_pred'] = y_test_pred
        nn[algo]['recall_test'] = recall_test
        
        print(f"\nRecall Score (TESTING): {recall_test}%\n")

    t3 = time()
    loopS = t3 - t2
    loopM = loopS / 60
    print(f'Completed Fitting and Predicting\nTime (Seconds): {loopS}\nTime (Minutes): {loopM}')

    return nn

def main():

    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')

    # nn = good_weights(X_train_scaled_cd, y_train_cd,
    #                   X_test_scaled_cd, y_test_cd)

if __name__ == "__main__":
    main()