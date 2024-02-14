# load and preprocess
from wrangle import final_dataset

# manipulate data
import numpy as np

# visualize data
import matplotlib.pyplot as plt

# machine learning models
import mlrose_hiive as rh

# model evaluation
from sklearn.metrics import (balanced_accuracy_score,
                             recall_score)

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
            
            nn = rh.NeuralNetwork(hidden_nodes=[4],
                                  activation='relu',
                                  algorithm='random_hill_climb',
                                  clip_max=1,
                                  restarts=200,
                                  max_iters=300,
                                  random_state=123,
                                  curve=True)
            
        case 'Simulated Annealing':
            print(f"Running {algo}")
            
            nn = rh.NeuralNetwork(hidden_nodes=[4],
                                  activation='relu',
                                  algorithm='simulated_annealing',
                                  clip_max=1,
                                  schedule=rh.GeomDecay(), 
                                  max_iters=300,
                                  random_state=123,
                                  curve = True)
            
        case 'Genetic Algorithm':
            print(f"Running {algo}")
            
            nn = rh.NeuralNetwork(hidden_nodes=[4], 
                                  activation='relu',
                                  algorithm='genetic_alg', 
                                  clip_max=1, 
                                  pop_size=200,
                                  mutation_prob=0.1, 
                                  max_iters=300, 
                                  random_state=123, 
                                  curve=True)            
            
    return nn

def good_weights(X_train, y_train, X_test, y_test):
    """Uses selected algorithm to search for optimal NN weights

    Args:
        data (pd.DataFrame): data to use for optimization
    """

    algos = np.array(['Random Hill Climbing',
                      'Simulated Annealing',
                      'Genetic Algorithm',])

    print('Running All Algorithms')
    nn = {algo: {'nn': get_nn_algo(algo=algo)} for algo in algos}
    print('Successfully Run All Algorithms')

    for algo in algos:
        print(f'\nWorking on: {algo}\nFitting: {algo}')
        nn[algo]['nn'].fit(X_train, y_train)

        # TRAINING
        print(f'\nPredicting (Training): {algo}')
        y_train_pred = nn[algo]['nn'].predict(X_train)

        print(f'\nCalculating Recall (Training): {algo}')
        recall_train = recall_score(y_train, y_train_pred)

        print(f'\nAppending Results (Training): {algo}')
        nn[algo]['y_train_pred'] = y_train_pred
        nn[algo]['recall_train'] = recall_train
        
        print(f"\nThe train recall score: {recall_train}%")

        # TESTING
        print(f'\nPredicting (Testing): {algo}')
        y_test_pred = nn[algo]['nn'].predict(X_test)

        print(f'\nCalculting Recall (Testing): {algo}')
        recall_test = recall_score(y_test, y_test_pred)

        print(f'\nAppending Results (Testing): {algo}')
        nn[algo]['y_test_pred'] = y_test_pred
        nn[algo]['recall_test'] = recall_test
        
        print(f"\nThe test recall score: {recall_test}%")

        return nn

def get_op_algo(algo):
    pass

def good_problem(algo):

    algos = np.array(['Random Hill Climbing'
                      'Simulated Annealing',
                      'Genetic Algorithm',
                      'mimic'])

    problems = np.array([])

    print('Running All Algorithms')
    opt_prob = {algo: {problem: get_op_algo(algo=algo)} for algo in algos for problem in problems}
    print('Successfully Run All Algorithms')

    return op
    pass

def main():

    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')

    # nn = good_weights(X_train_scaled_cd, y_train_cd,
    #                  X_test_scaled_cd, y_test_cd)

    # op = algo_opt_problem()

if __name__ == "__main__":
    main()