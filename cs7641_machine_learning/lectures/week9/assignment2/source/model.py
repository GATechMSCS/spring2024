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
                             make_scorer,
                             recall_score,
                             log_loss)

np.random.seed(123)

def choose_weights_optimizer(choice):

    match choice:
        case '1':
            print("You selected: Random Restart Hill Climbing")
            neural = ml.NeuralNetwork([67],
                                      algorithm=
                                      'random_hill_climb',
                                      clip_max=1,
                                      restarts=200,
                                      max_iters=1000,
                                      random_state=0,
                                      curve=True)
            
        case '2':
            print("You selected: Simulated Annealing")
            neural = ml.NeuralNetwork([67],
                                      algorithm='simulated_annealing',
                                      clip_max=1,
                                      schedule=ml.GeomDecay(), 
                                      max_iters=1000,
                                      random_state=123,
                                      curve = True)
            
        case '3':
            print("You selected: Genetic Algorithm")
            neural = ml.NeuralNetwork([67], 
                                      algorithm='genetic_alg', 
                                      clip_max=1, 
                                      pop_size=200,
                                      mutation_prob=0.1, 
                                      max_iters=1000, 
                                      random_state=0, 
                                      curve=True)
            
    return neural

def optimize_weights(data):
    """Uses selected algorithm to search for optimal NN weights

    Args:
        data (pd.DataFrame): data to use for optimization
    """

    choices = np.array([1, 2 , 3])
    
    neural = [choose_weights_optimizer(choice=choice) for choice in choices]

    # Fit weights to training data using chosen optimization alg.
    neural.fit(data[0], data[2])
    y_train_preds = neural.predict(data[0])
    train_accuracy = f1_score(data[2], y_train_preds)
    print("The F1-score training accuracy of the neural network was: " + "{:.2f}".format(train_accuracy) + "%")
    
    y_test_preds = neural.predict(data[1])
    test_accuracy = f1_score(data[3], y_test_preds)
    print("The F1-score test accuracy of the neural network was: " + "{:.2f}".format(test_accuracy) + "%")
    
    # Create/display fitness curve plot for selected algorithm.
    plt.plot(neural.fitness_curve)
    plt.ylabel('Relative Fitness Found')
    plt.show()

def main():

    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')

    X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')

    #optimize_weights(data)

if __name__ == "__main__":
    main()