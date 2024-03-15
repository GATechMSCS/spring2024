# load, preprocess, scale, baseline
from wrangle import final_dataset

# manipulate data
import pandas as pd
import numpy as np

# model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (make_scorer, log_loss)

# plotting
import matplotlib.pyplot as plt
from sklearn.model_selection import (learning_curve, LearningCurveDisplay,
                                    validation_curve, ValidationCurveDisplay)

np.random.seed(123)

home_loc = f"/home/leonardo_leads/Documents/SchoolDocs/"
gt_loc = f'{home_loc}ga_tech_masters/omscs_ml/spring2024/'
course_assign = f'{gt_loc}cs7641_machine_learning/lectures/week12/assignment3/source/'

def learn_curve_display(X_train, y_train, dset, sizes, model_name):

    step = 'step5' if model_name in ['gm', 'km'] else 'step4'

    if dset == 'cvd':
        scoring = 'recall'

    elif dset == 'nf':
        scoring = 'balanced_accuracy'

    match model_name:
        case 'pca':
            estimator = MLPClassifier(activation='logistic',
                                        alpha=0.0001,
                                        batch_size=200,
                                        hidden_layer_sizes=(250,),
                                        learning_rate_init=0.1,
                                        max_iter=200,
                                        random_state=123)

        case 'ica':
            estimator = MLPClassifier(activation='logistic',
                                        alpha=0.001,
                                        batch_size=250,
                                        hidden_layer_sizes=(150,),
                                        learning_rate_init=0.01,
                                        max_iter=200,
                                        random_state=123)

        case 'srp':
            estimator = MLPClassifier(activation='logistic',
                                        alpha=0.01,
                                        batch_size=200,
                                        hidden_layer_sizes=(250,),
                                        learning_rate_init=0.01,
                                        max_iter=200,
                                        random_state=123)

        case 'hlle':
            estimator = MLPClassifier(activation='relu',
                                        alpha=0.01,
                                        batch_size=150,
                                        hidden_layer_sizes=(150,),
                                        learning_rate_init=0.1,
                                        max_iter=200,
                                        random_state=123)

        case 'gm':
            estimator = MLPClassifier(activation='relu',
                                        alpha=0.01,
                                        batch_size=150,
                                        hidden_layer_sizes=(150,),
                                        learning_rate_init=0.1,
                                        max_iter=200,
                                        random_state=123)
            
        case 'km':
            estimator = MLPClassifier(activation='relu',
                                        alpha=0.01,
                                        batch_size=150,
                                        hidden_layer_sizes=(150,),
                                        learning_rate_init=0.1,
                                        max_iter=200,
                                        random_state=123)

    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(estimator=estimator,
                                                                                        X=X_train,
                                                                                        y=y_train,
                                                                                        train_sizes=sizes,
                                                                                        cv=4,
                                                                                        scoring=scoring,
                                                                                        n_jobs=-1,
                                                                                        random_state=123,
                                                                                        return_times=True)

    LearningCurveDisplay(train_sizes=train_sizes,
                        train_scores=train_scores,
                        test_scores=test_scores,
                        score_name=scoring).plot()
    plt.title(label=f"{model_name.upper()} Learning Curve (Training Sizes)")
    plt.legend(['Train Score', 'Validation Score'])
    save_loc = f'{course_assign}plots/{dset}/nn/{step}/{model_name}/training_sizes.png'
    plt.savefig(fname=f"{save_loc}")

    return fit_times, score_times

def val_curve_display(X_train, y_train, dset, param_name, param_range, model_name, error=False):
    
    scoring = 'recall' if dset == 'cvd' else 'balance_accuracy'
    
    score_name = scoring if error == False else 'Log Loss'
    
    step = 'step5' if model_name in ['gm', 'km'] else 'step4'
    
    if error:
        scoring = make_scorer(score_func=log_loss, 
                            response_method='predict_proba')
        param_name = 'max_iter'
        param_range = [100, 125, 150, 175, 200]

    match model_name:
        case 'pca':
            estimator = MLPClassifier(activation='logistic',
                                        alpha=0.0001,
                                        batch_size=200,
                                        hidden_layer_sizes=(250,),
                                        learning_rate_init=0.1,
                                        max_iter=200,
                                        random_state=123)

        case 'ica':
            estimator = MLPClassifier(activation='logistic',
                                        alpha=0.001,
                                        batch_size=250,
                                        hidden_layer_sizes=(150,),
                                        learning_rate_init=0.01,
                                        max_iter=200,
                                        random_state=123)

        case 'srp':
            estimator = MLPClassifier(activation='logistic',
                                        alpha=0.01,
                                        batch_size=200,
                                        hidden_layer_sizes=(250,),
                                        learning_rate_init=0.01,
                                        max_iter=200,
                                        random_state=123)

        case 'hlle':
            estimator = MLPClassifier(activation='relu',
                                        alpha=0.01,
                                        batch_size=150,
                                        hidden_layer_sizes=(150,),
                                        learning_rate_init=0.1,
                                        max_iter=200,
                                        random_state=123)

        case 'gm':
            estimator = MLPClassifier(activation='relu',
                                        alpha=0.01,
                                        batch_size=150,
                                        hidden_layer_sizes=(150,),
                                        learning_rate_init=0.1,
                                        max_iter=200,
                                        random_state=123)
            
        case 'km':
            estimator = MLPClassifier(activation='relu',
                                        alpha=0.01,
                                        batch_size=150,
                                        hidden_layer_sizes=(150,),
                                        learning_rate_init=0.1,
                                        max_iter=200,
                                        random_state=123)  

    train_scores, test_scores = validation_curve(estimator=estimator,
                                                    X=X_train,
                                                    y=y_train,
                                                    param_name=param_name,
                                                    param_range=param_range,
                                                    cv=4,
                                                    scoring=scoring,
                                                    n_jobs=-1,)

    ValidationCurveDisplay(param_name=param_name,
                            param_range=param_range,
                            train_scores=train_scores,
                            test_scores=test_scores,
                            score_name=score_name).plot()
    plt.title(label=f"{model_name.upper()} Validation Curve ({param_name})")
    plt.legend(['Train Score', 'Validation Score'])
    save_loc = f'{course_assign}plots/{dset}/nn/{step}/{model_name}/{param_name}.png'
    plt.savefig(fname=f"{save_loc}")

def main():
    
    cd_params = {'param1_name': 'batch_size',
                'param1_range': [50, 55, 60, 65, 70],
                'param2_name': 'hidden_layer_sizes',
                'param2_range': [50, 75, 100, 125, 150],
                'sizes': [0.7, 0.75, 0.80, 0.90]}

    # nf_params = {'param1_name': 'batch_size',
    #             'param1_range': [25, 50, 75, 100, 125],
    #             'param2_name': 'hidden_layer_sizes',
    #             'param2_range': [125, 150, 175, 200, 225, 230],
    #             'sizes': [0.7, 0.75, 0.80, 0.90]}

    # CVD
    X_train_scaled_cd, X_test_scaled_cd, y_train_cd, y_test_cd = final_dataset(dataset='cvd')

    # Plot CVD Model
    for model in ['gm', 'km']:# ['pca', 'ica', 'srp', 'hlle', 'gm' 'km']:
        fit_times, score_times = learn_curve_display(X_train_scaled_cd,
                                                    y_train_cd,
                                                    dset='cvd',
                                                    sizes=cd_params['sizes'],
                                                    model_name=model)

        val_curve_display(X_train_scaled_cd, y_train_cd, 'cvd', 
                            cd_params['param1_name'], cd_params['param1_range'], 
                            model_name=model, error=False)
        val_curve_display(X_train_scaled_cd, y_train_cd, 'cvd', 
                            cd_params['param2_name'], cd_params['param2_range'], 
                            model_name=model, error=False)
        val_curve_display(X_train_scaled_cd, y_train_cd, 'cvd',
                            None, None, model_name=model, error=True)

    # NF
    # X_train_scaled_nf, X_test_scaled_nf, y_train_nf, y_test_nf = final_dataset(dataset='nf')

    # Plot NF Model
    # fit_times, score_times = learn_curve_display(X_train_scaled_nf,
    #                                             y_train_nf,
    #                                             dset='nf',
    #                                             sizes=nf_params['sizes'])

    # val_curve_display(X_train_scaled_nf, y_train_nf, 'nf', nf_params['param1_name'], nf_params['param1_range'], error=False)
    # val_curve_display(X_train_scaled_nf, y_train_nf, 'nf', nf_params['param2_name'], nf_params['param2_range'], error=False)
    # val_curve_display(X_train_scaled_nf, y_train_nf, 'nf', None, None, error=True)

if __name__ == "__main__":
    main()