# load and preprocess
from wrangle import final_dataset

# manipulate data
import numpy as np

# seed and size combos
from seed_size_combos import algos_sps

# machine learning models
import mlrose_hiive as rh

# timing
from time import time

np.random.seed(123)

def get_op_algo(algo:str, seed:int,
                lengthFP:int,
                lengthFF:int,
                max_item_countKS:int):

    match algo:

        case 'Random Hill Climbing':
            print(f"\nRunning {algo}")

            # Four Peaks
            print(f'Running {algo} FourPeaks')
            t0 = time()
            fitness_fnc_FP = rh.FourPeaks(t_pct=0.1)
            op_type_FP = rh.DiscreteOpt(length=lengthFP,
                                        fitness_fn=fitness_fnc_FP,
                                        maximize=True,
                                        max_val=2)
            
            df_run_stats_FP, df_run_curves_FP = rh.RHCRunner(problem=op_type_FP,
                                                             experiment_name='rhc1',
                                                             iteration_list=2**np.arange(10),
                                                             restart_list=[10, 20, 30],
                                                             seed=seed,).run()
            t1 = time()
            seconds1 = t1 - t0
            minutes1 = seconds1 / 60
            print(f'Completed {algo} FourPeaks. Time: {minutes1} minutes.')
            
            # FlipFlop
            print(f'Running {algo} FlipFlop')
            t2 = time()
            fitness_fnc_FF = rh.FlipFlop()
            op_type_FF = rh.DiscreteOpt(length=lengthFF,
                                        fitness_fn=fitness_fnc_FF,
                                        maximize=True,
                                        max_val=2)
            
            df_run_stats_FF, df_run_curves_FF = rh.RHCRunner(problem=op_type_FF,
                                                             experiment_name='rhc1',
                                                             iteration_list=2**np.arange(10),
                                                             restart_list=[10, 20, 30],
                                                             seed=seed,).run()
            t3 = time()
            seconds2 = t3 - t2
            minutes2 = seconds2 / 60
            print(f'Completed {algo} FlipFlop. Time: {minutes2} minutes.')
            
            # Knapsack
            print(f'Running {algo} Knapsack')
            t5 = time()
            fitness_fnc_KS = rh.Knapsack(max_item_count=max_item_countKS,
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
                                                             seed=seed).run()
            t6 = time()
            seconds3 = t6 - t5
            minutes3 = seconds3 / 60
            print(f'Completed {algo} Knapsack. Time: {minutes3} minutes.')
            print(f"Completed {algo}. Time: {minutes3 + minutes2 + minutes1} minutes")
            
        case 'Simulated Annealing':
            print(f"\nRunning {algo}")

            # Four Peaks
            print(f'Running {algo} FourPeaks')
            t0 = time()
            fitness_fnc_FP = rh.FourPeaks(t_pct=0.1)
            op_type_FP = rh.DiscreteOpt(length=lengthFP,
                                        fitness_fn=fitness_fnc_FP,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_FP, df_run_curves_FP = rh.SARunner(problem=op_type_FP,
                                                            experiment_name='sa1',
                                                            iteration_list=2**np.arange(10),
                                                            seed=seed,
                                                            temperature_list=[1, 10, 50, 100, 200, 250],
                                                            decay_list=[rh.GeomDecay]).run()
            t1 = time()
            seconds1 = t1 - t0
            minutes1 = seconds1 / 60
            print(f'Completed {algo} FourPeaks. Time: {minutes1} minutes.')

            # FlipFlop
            print(f'Running {algo} FlipFlop')
            t2 = time()
            fitness_fnc_FF = rh.FlipFlop()
            op_type_FF = rh.DiscreteOpt(length=lengthFF,
                                        fitness_fn=fitness_fnc_FF,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_FF, df_run_curves_FF = rh.SARunner(problem=op_type_FF,
                                                            experiment_name='sa1',
                                                            iteration_list=2**np.arange(10),
                                                            seed=seed,
                                                            temperature_list=[1, 10, 50, 100, 200, 250],
                                                            decay_list=[rh.GeomDecay]).run()
            t3 = time()
            seconds2 = t3 - t2
            minutes2 = seconds2 / 60
            print(f'Completed {algo} FlipFlop. Time: {minutes2} minutes.')

            # Knapsack
            print(f'Running {algo} Knapsack')
            t5 = time()
            fitness_fnc_KS = rh.Knapsack(max_item_count=max_item_countKS,
                                         weights=[0.25, 0.50, 0.75, 1.0],
                                         values=[2.5, 5.0, 7.5, 10])
            op_type_KS = rh.DiscreteOpt(length=4,
                                        fitness_fn=fitness_fnc_KS,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_KS, df_run_curves_KS = rh.SARunner(problem=op_type_KS,
                                                            experiment_name='sa1',
                                                            iteration_list=2**np.arange(10),
                                                            seed=seed,
                                                            temperature_list=[1, 10, 50, 100, 200, 250],
                                                            decay_list=[rh.GeomDecay]).run()
            t6 = time()
            seconds3 = t6 - t5
            minutes3 = seconds3 / 60
            print(f'Completed {algo} Knapsack. Time: {minutes3} minutes.')
            print(f"Completed {algo}. Time: {minutes3 + minutes2 + minutes1} minutes")

        case 'Genetic Algorithm':
            print(f"\nRunning {algo}")

            # Four Peaks
            print(f'Running {algo} FourPeaks')
            t0 = time()
            fitness_fnc_FP = rh.FourPeaks(t_pct=0.1)
            op_type_FP = rh.DiscreteOpt(length=lengthFP,
                                        fitness_fn=fitness_fnc_FP,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_FP, df_run_curves_FP = rh.GARunner(problem=op_type_FP,
                                                            experiment_name='GA1',
                                                            seed=seed,
                                                            iteration_list=2**np.arange(10),
                                                            population_sizes=[100, 150, 200],
                                                            mutation_rates=[0.4, 0.5, 0.6]).run()
            t1 = time()
            seconds1 = t1 - t0
            minutes1 = seconds1 / 60
            print(f'Completed {algo} FourPeaks. Time: {minutes1} minutes.')

            # FlipFlop
            print(f'Running {algo} FlipFlop')
            t2 = time()
            fitness_fnc_FF = rh.FlipFlop()
            op_type_FF = rh.DiscreteOpt(length=lengthFF,
                                        fitness_fn=fitness_fnc_FF,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_FF, df_run_curves_FF = rh.GARunner(problem=op_type_FF,
                                                            experiment_name='GA1',
                                                            seed=seed,
                                                            iteration_list=2**np.arange(10),
                                                            population_sizes=[100, 150, 200],
                                                            mutation_rates=[0.4, 0.5, 0.6]).run()
            t3 = time()
            seconds2 = t3 - t2
            minutes2 = seconds2 / 60
            print(f'Completed {algo} FlipFlop. Time: {minutes2} minutes.')

            # Knapsack
            print(f'Running {algo} Knapsack')
            t5 = time()
            fitness_fnc_KS = rh.Knapsack(max_item_count=max_item_countKS,
                                         weights=[0.25, 0.50, 0.75, 1.0],
                                         values=[2.5, 5.0, 7.5, 10])
            op_type_KS = rh.DiscreteOpt(length=4,
                                        fitness_fn=fitness_fnc_KS,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_KS, df_run_curves_KS = rh.GARunner(problem=op_type_KS,
                                                            experiment_name='GA1',
                                                            seed=seed,
                                                            iteration_list=2**np.arange(10),
                                                            population_sizes=[100, 150, 200],
                                                            mutation_rates=[0.4, 0.5, 0.6]).run()
            t6 = time()
            seconds3 = t6 - t5
            minutes3 = seconds3 / 60
            print(f'Completed {algo} Knapsack. Time: {minutes3} minutes.')
            print(f"Completed {algo}. Time: {minutes3 + minutes2 + minutes1} minutes")

        case 'MIMIC':
            print(f"\nRunning {algo}")

            # Four Peaks
            print(f'Running {algo} FourPeaks')
            t0 = time()
            fitness_fnc_FP = rh.FourPeaks(t_pct=0.1)
            op_type_FP = rh.DiscreteOpt(length=lengthFP, 
                                        fitness_fn=fitness_fnc_FP,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_FP, df_run_curves_FP = rh.MIMICRunner(problem=op_type_FP,
                                                               experiment_name='MIMIC1',
                                                               seed=seed,
                                                               iteration_list=2**np.arange(10),
                                                               keep_percent_list=[0.25, 0.5, 0.75],
                                                               population_sizes=[150, 200, 250],
                                                               fast_mimic=True).run()
            t1 = time()
            seconds1 = t1 - t0
            minutes1 = seconds1 / 60            
            print(f'Completed {algo} FourPeaks. Time: {minutes1} minutes.')

            # FlipFlop
            print(f'Running {algo} FlipFlop')
            t2 = time()            
            fitness_fnc_FF = rh.FlipFlop()
            op_type_FF = rh.DiscreteOpt(length=lengthFF, 
                                        fitness_fn=fitness_fnc_FF,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_FF, df_run_curves_FF = rh.MIMICRunner(problem=op_type_FF,
                                                               experiment_name='MIMIC1',
                                                               seed=seed,
                                                               iteration_list=2**np.arange(10),
                                                               keep_percent_list=[0.25, 0.5, 0.75],
                                                               population_sizes=[150, 200, 250],
                                                               fast_mimic=True).run()
            t3 = time()
            seconds2 = t3 - t2
            minutes2 = seconds2 / 60
            print(f'Completed {algo} FlipFlop. Time: {minutes2} minutes.')

            # Knapsack
            print(f'Running {algo} Knapsack')
            t5 = time()            
            fitness_fnc_KS = rh.Knapsack(max_item_count=max_item_countKS,
                                         weights=[0.25, 0.50, 0.75, 1.0],
                                         values=[2.5, 5.0, 7.5, 10])
            op_type_KS = rh.DiscreteOpt(length=4, 
                                        fitness_fn=fitness_fnc_KS,
                                        maximize=True,
                                        max_val=2)

            df_run_stats_KS, df_run_curves_KS = rh.MIMICRunner(problem=op_type_KS,
                                                               experiment_name='MIMIC1',
                                                               seed=seed,
                                                               iteration_list=2**np.arange(10),
                                                               keep_percent_list=[0.25, 0.5, 0.75],
                                                               population_sizes=[150, 200, 250],
                                                               fast_mimic=True).run()
            t6 = time()
            seconds3 = t6 - t5
            minutes3 = seconds3 / 60
            print(f'Completed {algo} Knapsack. Time: {minutes3} minutes.')
            print(f"Completed {algo}. Time: {minutes3 + minutes2 + minutes1} minutes")

    df_run_stats =  {'df_run_stats_FP': df_run_stats_FP,
                     'df_run_stats_FF': df_run_stats_FF,
                     'df_run_stats_KS': df_run_stats_KS}

    df_run_curves = {'df_run_curves_FP': df_run_curves_FP,
                     'df_run_curves_FF': df_run_curves_FF,
                     'df_run_curves_KS': df_run_curves_KS}
    
    return df_run_stats, df_run_curves

def good_problem():

    print('Running All Algorithms')
    t0 = time()
    for algo, combos in algos_sps.items():
        for combo, seed_size in combos.items():
            seed_size_keys = list(seed_size)
            seed = seed_size[seed_size_keys[0]]
            lengthFP =seed_size[seed_size_keys[1]]
            lengthFF = seed_size[seed_size_keys[2]]
            max_item_countKS = seed_size[seed_size_keys[3]]
            algos_sps[algo][combo]['results'] = get_op_algo(algo=algo,
                                                            seed=seed,
                                                            lengthFP=lengthFP,
                                                            lengthFF=lengthFF,
                                                            max_item_countKS=max_item_countKS)

    t1 = time()
    seconds = t1 - t0
    minutes = seconds / 60
    print(f'Compmleted All Algorithms\nTime (Seconds): {seconds}\nTime (Minutes): {minutes} ')

    return algos_sps

def main():

    opt_prob = good_problem()

if __name__ == "__main__":
    main()