import mlrose_hiive as rh

nn_algo_tune = \
{
'Random Hill Climbing': {'rhc_combo1': {'restarts': 5,
                                        'learning_rate': 0.01,},
                         'rhc_combo2': {'restarts': 10,
                                        'learning_rate': 0.05,},
                         'rhc_combo3': {'restarts': 15,
                                        'learning_rate': 0.1,},},
'Simulated Annealing': {'sa_combo1': {'schedule': rh.GeomDecay(init_temp=50.0,
                                                                decay=0.75,
                                                                min_temp=1.0),
                                      'learning_rate': 0.01,},
                        'sa_combo2': {'schedule': rh.ArithDecay(init_temp=50.0,
                                                                  decay=0.75,
                                                                  min_temp=0.001),
                                      'learning_rate': 0.05,},
                        'sa_combo3': {'schedule': rh.ExpDecay(init_temp=50.0,
                                                                exp_const=0.75,
                                                                min_temp=0.001),
                                      'learning_rate': 0.1,},},
'Genetic Algorithm': {'ga_combo1': {'pop_size': 175,
                                    'learning_rate': 0.01,
                                    'mutation_prob': 0.05,},
                      'ga_combo2': {'pop_size': 200,
                                    'learning_rate': 0.05,
                                    'mutation_prob': 0.1,},
                      'ga_combo3': {'pop_size': 225,
                                    'learning_rate': 0.1,
                                    'mutation_prob': 0.15,},},
'Gradient Descent': {'Default_Combo': {'No Hypers': None}}
}

def main():
    import numpy as np
    for algo, combos in nn_algo_tune.items():        
        for combo, hypers in combos.items():
            hypers_keys = list(hypers)

            match algo:
                case 'Random Hill Climbing':
                    rs = hypers[hypers_keys[0]]
                    lr = hypers[hypers_keys[1]]

                case 'Simulated Annealing':
                    dule = hypers[hypers_keys[0]]
                    dule_eval = dule.evaluate(4)
                    lr = hypers[hypers_keys[1]]
                    print(dule)

                case 'Genetic Algorithm':
                    ps = hypers[hypers_keys[0]]
                    lr = hypers[hypers_keys[1]]
                    mp = hypers[hypers_keys[2]]

if __name__ == "__main__":
    main()