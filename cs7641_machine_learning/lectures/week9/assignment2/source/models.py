# load and preprocess
from wrangle import final_dataset

# manipulate data
import numpy as np

# visualize data
import matplotlib.pyplot as plt

# machine learning models
import mlrose_hiive as rh

# model evaluation
from sklearn.metrics import recall_score

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
                                  curve=True,)
            
        case 'Simulated Annealing':
            print(f"Running {algo}")
            
            nn = rh.NeuralNetwork(hidden_nodes=[4],
                                  activation='relu',
                                  algorithm='simulated_annealing',
                                  clip_max=1,
                                  schedule=rh.GeomDecay(), 
                                  max_iters=300,
                                  random_state=123,
                                  curve = True,)
            
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
                                  curve=True,)

        case 'Gradient Descent':
            print(f"Running {algo}")
            
            nn = rh.NeuralNetwork(hidden_nodes=[4], 
                                  activation='relu',
                                  algorithm='gradient_descent', 
                                  clip_max=1, 
                                  pop_size=200, 
                                  max_iters=300, 
                                  random_state=123, 
                                  curve=True,)   
            
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
    nn = {algo: {'nn': get_nn_algo(algo=algo)} for algo in algos}
    print('Successfully Run All Algorithms')

    for algo in algos:
        print(f'\nWorking on: {algo}\nFitting: {algo}')
        nn[algo]['nn'].fit(X_train, y_train)

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

    return nn

def get_op_algo(algo):

    match algo:

        case 'Random Hill Climbing':
            print(f"Running {algo}")

            # Four Peaks
            fitness_fnc_FP = rh.FourPeaks()
            op_type_FP = rh.DiscreteOpt(length=4, fitness_fn=fitness_fnc_FP,
                                        maximize=True,
                                        max_val=2)
            
            best_state_FP, best_fitness_FP, fitness_curve_FP = rh.random_hill_climb(problem=op_type_FP,
                                                                                 random_state=123,
                                                                                 curve=True)

            # FlipFlop
            fitness_fnc_FF = rh.FlipFlop()
            op_type_FF = rh.DiscreteOpt(length=4, fitness_fn=fitness_fnc_FF,
                                       maximize=True,
                                       max_val=2)
            
            best_state_FF, best_fitness_FF, fitness_curve_FF = rh.random_hill_climb(problem=op_type_FF,
                                                                               random_state=123,
                                                                               curve=True)

            # Knapsack
            fitness_fnc_KS = rh.Knapsack(weights=[0.25, 0.50, 0.75, 1.0],
                                         values=[2.5, 5.0, 7.5, 10])
            op_type_KS = rh.DiscreteOpt(length=4, fitness_fn=fitness_fnc_KS,
                                         maximize=True,
                                         max_val=2)

            best_state_KS, best_fitness_KS, fitness_curve_KS = rh.random_hill_climb(problem=op_type_KS,
                                                                                   random_state=123,
                                                                                   curve=True)
            
        case 'Simulated Annealing':
            print(f"Running {algo}")

            # Four Peaks
            fitness_fnc_FP = rh.FourPeaks()
            op_type_FP = rh.DiscreteOpt(length=4, fitness_fn=fitness_fnc_FP,
                                        maximize=True,
                                        max_val=2)

            best_state_FP, best_fitness_FP, fitness_curve_FP = rh.simulated_annealing(problem=op_type_FP,
                                                                                   random_state=123, 
                                                                                   curve=True)

            # FlipFlop
            fitness_fnc_FF = rh.FlipFlop()
            op_type_FF = rh.DiscreteOpt(length=4, fitness_fn=fitness_fnc_FF,
                                       maximize=True,
                                       max_val=2)

            best_state_FF, best_fitness_FF, fitness_curve_FF = rh.simulated_annealing(problem=op_type_FF,
                                                                                random_state=123, 
                                                                                curve=True)
            
            # Knapsack
            fitness_fnc_KS = rh.Knapsack(weights=[0.25, 0.50, 0.75, 1.0],
                                         values=[2.5, 5.0, 7.5, 10])
            op_type_KS = rh.DiscreteOpt(length=4, fitness_fn=fitness_fnc_KS,
                                         maximize=True,
                                         max_val=2)

            best_state_KS, best_fitness_KS, fitness_curve_KS = rh.simulated_annealing(problem=op_type_KS,
                                                                                    random_state=123, 
                                                                                    curve=True)
            
        case 'Genetic Algorithm':
            print(f"Running {algo}")

            # Four Peaks
            fitness_fnc_FP = rh.FourPeaks()
            op_type_FP = rh.DiscreteOpt(length=4, fitness_fn=fitness_fnc_FP,
                                        maximize=True,
                                        max_val=2)

            best_state_FP, best_fitness_FP, fitness_curve_FP = rh.genetic_alg(problem=op_type_FP,
                                                                           random_state=123,
                                                                           curve=True)

            # FlipFlop
            fitness_fnc_FF = rh.FlipFlop()
            op_type_FF = rh.DiscreteOpt(length=4, fitness_fn=fitness_fnc_FF,
                                        maximize=True,
                                        max_val=2)

            best_state_FF, best_fitness_FF, fitness_curve_FF = rh.genetic_alg(problem=op_type_FF,
                                                                         random_state=123,
                                                                         curve=True)

            # Knapsack
            fitness_fnc_KS = rh.Knapsack(weights=[0.25, 0.50, 0.75, 1.0],
                                         values=[2.5, 5.0, 7.5, 10])
            op_type_KS = rh.DiscreteOpt(length=4, fitness_fn=fitness_fnc_KS,
                                        maximize=True,
                                        max_val=2)

            best_state_KS, best_fitness_KS, fitness_curve_KS = rh.genetic_alg(problem=op_type_KS,
                                                                             random_state=123,
                                                                             curve=True)

        case 'MIMIC':
            print(f"Running {algo}")

            # Four Peaks
            fitness_fnc_FP = rh.FourPeaks()
            op_type_FP = rh.DiscreteOpt(length=4, fitness_fn=fitness_fnc_FP,
                                        maximize=True,
                                        max_val=2)

            best_state_FP, best_fitness_FP, fitness_curve_FP = rh.mimic(problem=op_type_FP,
                                                                     random_state=123,
                                                                     curve=True)

            # FlipFlop
            fitness_fnc_FF = rh.FlipFlop()
            op_type_FF = rh.DiscreteOpt(length=4, fitness_fn=fitness_fnc_FF,
                                       maximize=True,
                                       max_val=2)

            best_state_FF, best_fitness_FF, fitness_curve_FF = rh.mimic(problem=op_type_FF,
                                                                   random_state=123,
                                                                   curve=True)

            # Knapsack
            fitness_fnc_KS = rh.Knapsack(weights=[0.25, 0.50, 0.75, 1.0],
                                         values=[2.5, 5.0, 7.5, 10])
            op_type_KS = rh.DiscreteOpt(length=4, fitness_fn=fitness_fnc_KS,
                                         maximize=True,
                                         max_val=2)

            best_state_KS, best_fitness_KS, fitness_curve_KS = rh.mimic(problem=op_type_KS,
                                                                       random_state=123,
                                                                       curve=True)
            
    best_state =  {'best_state_FP': best_state_FP,
                   'best_state_FF': best_state_FF,
                   'best_state_KS': best_state_KS}

    best_fitness = {'best_fitness_FP': best_fitness_FP,
                    'best_fitness_FF': best_fitness_FF,
                    'best_fitness_KS': best_fitness_KS}

    fitness_curve = {'fitness_curve_FP': fitness_curve_FP,
                     'fitness_curve_FF': fitness_curve_FF,
                     'fitness_curve_KS': fitness_curve_KS}
    
    return best_state, best_fitness, fitness_curve

def good_problem():

    algos = np.array(['Random Hill Climbing',
             'Simulated Annealing',
             'Genetic Algorithm',
             'MIMIC'])

    print('Running All Algorithms')
    opt_prob = {algo: get_op_algo(algo=algo) for algo in algos}
    print('Successfully Run All Algorithms')

    return opt_prob

def main():

    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')

    # nn = good_weights(X_train_scaled_cd, y_train_cd,
    #                   X_test_scaled_cd, y_test_cd)

    # opt_prob = good_problem()

if __name__ == "__main__":
    main()