# load and preprocess
from wrangle import final_dataset

# manipulate data
import numpy as np

# machine learning models
import mlrose_hiive as rh

# timing
from time import time

np.random.seed(123)

def get_op_algo(algo, seed, prob_size):

    match algo:

        case 'Random Hill Climbing':
            print(f"\nRunning {algo}")

            # Four Peaks
            print(f'Running {algo} FourPeaks')
            fitness_fnc_FP = rh.FourPeaks(t_pct=0.1)
            op_type_FP = rh.DiscreteOpt(length=4,
                                        fitness_fn=fitness_fnc_FP,
                                        maximize=True,
                                        max_val=2)
            
            df_run_stats_FP, df_run_curves_FP = rh.RHCRunner(problem=op_type_FP,
                                                             experiment_name='rhc1',
                                                             iteration_list=2**np.arange(10),
                                                             restart_list=[10, 20, 30],
                                                             seed=123,).run()
            print(f'Completed {algo} FourPeaks')
            
            # FlipFlop
            print(f'Running {algo} FlipFlop')
            fitness_fnc_FF = rh.FlipFlop()
            op_type_FF = rh.DiscreteOpt(length=4,
                                        fitness_fn=fitness_fnc_FF,
                                        maximize=True,
                                        max_val=2)
            
            df_run_stats_FF, df_run_curves_FF = rh.RHCRunner(problem=op_type_FF,
                                                             experiment_name='rhc1',
                                                             iteration_list=2**np.arange(10),
                                                             restart_list=[10, 20, 30],
                                                             seed=123,).run()
            print(f'Completed {algo} FlipFlop')
            
            # Knapsack
            print(f'Running {algo} Knapsack')
            fitness_fnc_KS = rh.Knapsack(max_item_count=5,
                                         weights=[0.25, 0.50, 0.75, 1.0],
                                         values=[2.5, 5.0, 7.5, 10])
            op_type_KS = rh.DiscreteOpt(length=4,
                                        fitness_fn=fitness_fnc_KS,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_KS, df_run_curves_KS = rh.RHCRunner(problem=op_type_KS,
                                                             experiment_name='rhc1',
                                                             iteration_list=2**np.arange(10),
                                                             restart_list=[10, 20, 30],
                                                             seed=123).run()
            print(f'Completed {algo} Knapsack')
            print(f"Completed {algo}")
            
        case 'Simulated Annealing':
            print(f"\nRunning {algo}")

            # Four Peaks
            print(f'Running {algo} FourPeaks')
            fitness_fnc_FP = rh.FourPeaks(t_pct=0.1)
            op_type_FP = rh.DiscreteOpt(length=4,
                                        fitness_fn=fitness_fnc_FP,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_FP, df_run_curves_FP = rh.SARunner(problem=op_type_FP,
                                                            experiment_name='sa1',
                                                            iteration_list=2**np.arange(10),
                                                            seed=123,
                                                            temperature_list=[1, 10, 50, 100, 200, 250],
                                                            decay_list=[rh.GeomDecay]).run()
            print(f'Completed {algo} FourPeaks')

            # FlipFlop
            print(f'Running {algo} FlipFlop')
            fitness_fnc_FF = rh.FlipFlop()
            op_type_FF = rh.DiscreteOpt(length=4,
                                        fitness_fn=fitness_fnc_FF,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_FF, df_run_curves_FF = rh.SARunner(problem=op_type_FF,
                                                            experiment_name='sa1',
                                                            iteration_list=2**np.arange(10),
                                                            seed=123,
                                                            temperature_list=[1, 10, 50, 100, 200, 250],
                                                            decay_list=[rh.GeomDecay]).run()
            print(f'Completed {algo} FlipFlop')

            # Knapsack
            print(f'Running {algo} Knapsack')
            fitness_fnc_KS = rh.Knapsack(max_item_count=5,
                                         weights=[0.25, 0.50, 0.75, 1.0],
                                         values=[2.5, 5.0, 7.5, 10])
            op_type_KS = rh.DiscreteOpt(length=4,
                                        fitness_fn=fitness_fnc_KS,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_KS, df_run_curves_KS = rh.SARunner(problem=op_type_KS,
                                                            experiment_name='sa1',
                                                            iteration_list=2**np.arange(10),
                                                            seed=123,
                                                            temperature_list=[1, 10, 50, 100, 200, 250],
                                                            decay_list=[rh.GeomDecay]).run()
            print(f'Completed {algo} Knapsack')
            print(f"Completed {algo}")

        case 'Genetic Algorithm':
            print(f"\nRunning {algo}")

            # Four Peaks
            print(f'Running {algo} FourPeaks')
            fitness_fnc_FP = rh.FourPeaks(t_pct=0.1)
            op_type_FP = rh.DiscreteOpt(length=4,
                                        fitness_fn=fitness_fnc_FP,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_FP, df_run_curves_FP = rh.GARunner(problem=op_type_FP,
                                                            experiment_name='GA1',
                                                            seed=123,
                                                            iteration_list=2**np.arange(10),
                                                            population_sizes=[100, 150, 200],
                                                            mutation_rates=[0.4, 0.5, 0.6]).run()
            print(f'Completed {algo} FourPeaks')

            # FlipFlop
            print(f'Running {algo} FlipFlop')
            fitness_fnc_FF = rh.FlipFlop()
            op_type_FF = rh.DiscreteOpt(length=4,
                                        fitness_fn=fitness_fnc_FF,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_FF, df_run_curves_FF = rh.GARunner(problem=op_type_FF,
                                                            experiment_name='GA1',
                                                            seed=123,
                                                            iteration_list=2**np.arange(10),
                                                            population_sizes=[100, 150, 200],
                                                            mutation_rates=[0.4, 0.5, 0.6]).run()
            print(f'Completed {algo} FlipFlop')

            # Knapsack
            print(f'Running {algo} Knapsack')
            fitness_fnc_KS = rh.Knapsack(max_item_count=5,
                                         weights=[0.25, 0.50, 0.75, 1.0],
                                         values=[2.5, 5.0, 7.5, 10])
            op_type_KS = rh.DiscreteOpt(length=4,
                                        fitness_fn=fitness_fnc_KS,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_KS, df_run_curves_KS = rh.GARunner(problem=op_type_KS,
                                                            experiment_name='GA1',
                                                            seed=123,
                                                            iteration_list=2**np.arange(10),
                                                            population_sizes=[100, 150, 200],
                                                            mutation_rates=[0.4, 0.5, 0.6]).run()
            print(f'Completed {algo} Knapsack')
            print(f"Completed {algo}")        

        case 'MIMIC':
            print(f"\nRunning {algo}")

            # Four Peaks
            print(f'Running {algo} FourPeaks')
            fitness_fnc_FP = rh.FourPeaks(t_pct=0.1)
            op_type_FP = rh.DiscreteOpt(length=4, 
                                        fitness_fn=fitness_fnc_FP,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_FP, df_run_curves_FP = rh.MIMICRunner(problem=op_type_FP,
                                                               experiment_name='MIMIC1',
                                                               seed=123,
                                                               iteration_list=2**np.arange(10),
                                                               keep_percent_list=[0.25, 0.5, 0.75],
                                                               population_sizes=[150, 200, 250]).run()
            print(f'Completed {algo} FourPeaks')

            # FlipFlop
            print(f'Running {algo} FlipFlop')
            fitness_fnc_FF = rh.FlipFlop()
            op_type_FF = rh.DiscreteOpt(length=4, 
                                        fitness_fn=fitness_fnc_FF,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_FF, df_run_curves_FF = rh.MIMICRunner(problem=op_type_FF,
                                                               experiment_name='MIMIC1',
                                                               seed=123,
                                                               iteration_list=2**np.arange(10),
                                                               keep_percent_list=[0.25, 0.5, 0.75],
                                                               population_sizes=[150, 200, 250]).run()
            print(f'Completed {algo} FlipFlop')

            # Knapsack
            print(f'Running {algo} Knapsack')
            fitness_fnc_KS = rh.Knapsack(max_item_count=5,
                                         weights=[0.25, 0.50, 0.75, 1.0],
                                         values=[2.5, 5.0, 7.5, 10])
            op_type_KS = rh.DiscreteOpt(length=4, 
                                        fitness_fn=fitness_fnc_KS,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_KS, df_run_curves_KS = rh.MIMICRunner(problem=op_type_KS,
                                                               experiment_name='MIMIC1',
                                                               seed=123,
                                                               iteration_list=2**np.arange(10),
                                                               keep_percent_list=[0.25, 0.5, 0.75],
                                                               population_sizes=[150, 200, 250]).run()
            print(f'Completed {algo} Knapsack')
            print(f"Completed {algo}")

    # best_state =  {'best_state_FP': best_state_FP,
    #                'best_state_FF': best_state_FF,
    #                'best_state_KS': best_state_KS}

    # best_fitness = {'best_fitness_FP': best_fitness_FP,
    #                 'best_fitness_FF': best_fitness_FF,
    #                 'best_fitness_KS': best_fitness_KS}

    # fitness_curve = {'fitness_curve_FP': fitness_curve_FP,
    #                  'fitness_curve_FF': fitness_curve_FF,
    #                  'fitness_curve_KS': fitness_curve_KS}
    
    return 1

def good_problem():

    seed_pop_size = {'seed': [1, 2, 3],
                     'lengthFP': [4, 6, 8],
                     'lengthFF': [4, 6, 8],
                     'max_item_countKS': [6, 8,10]}

    algos = np.array(['Random Hill Climbing',
                      'Simulated Annealing',
                      'Genetic Algorithm',
                      'MIMIC'])

    print('Running All Algorithms')
    t0 = time()
    opt_prob = {algo: get_op_algo(algo=algo) for algo in algos}
    t1 = time()
    seconds = t1 - t0
    minutes = seconds / 60
    print(f'Compmleted All Algorithms\nTime (Seconds): {seconds}\nTime (Minutes): {minutes} ')

    return opt_prob

def main():

    opt_prob = good_problem()

if __name__ == "__main__":
    main()