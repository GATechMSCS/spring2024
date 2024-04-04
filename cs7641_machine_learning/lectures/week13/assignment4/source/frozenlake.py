import gymnasium as gym
from bettermdptools.utils.test_env import TestEnv
from bettermdptools.algorithms.rl import RL
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots
from bettermdptools.utils.blackjack_wrapper import BlackjackWrapper
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bettermdptools.utils.grid_search import GridSearch
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

np.random.seed(123)

def make_env(mdp, size):
    env = gym.make(id=mdp,
                   desc=size,
                   render_mode='ansi')

    return env

def run_q_gridsearch(process, gamma, epsilon_decay, iters):

    q_learning_results = GridSearch.q_learning_grid_search(env=process, gamma=gamma, epsilon_decay=epsilon_decay, iters=iters)    

    return q_learning_results

def run_pi_gs(process, gamma, epsilon_decay, iters):
    
    pi_grid_results = GridSearch.pi_grid_search(env=process, gamma=gamma, n_iters=iters, theta=epsilon_decay)

    return pi_grid_results

def run_vi_gs(process, gamma, epsilon_decay, iters):

    vi_grid_results = GridSearch.vi_grid_search(env=process, gamma=gamma, n_iters=iters, theta=epsilon_decay)

    return vi_grid_results

#test policy
# test_scores = TestEnv.test_env(env=frozenlake, n_iters=100, render=False, pi=pi_pi, user_input=False)
# print(np.mean(test_scores))

# FROM PLOTS NOTEBOOK
# state values heatmap
# frozen_lake = gym.make('FrozenLake8x8-v1', render_mode=None)
# V, V_track, pi = Planner(frozen_lake.P).value_iteration(n_iters=5000)
# size=(8,8)
# Plots.values_heat_map(V, "Frozen Lake\nValue Iteration State Values", size)

# state values vs iterations
# Clip trailing zeros in case convergence is reached before max iterations
# This is likely when setting the n_iters parameter
# max_value_per_iter = np.trim_zeros(np.mean(V_track, axis=1), 'b')
# Plots.v_iters_plot(max_value_per_iter, "Frozen Lake\nMean Value v Iterations")

# policy maps
# fl_actions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
# fl_map_size=(8,8)
# title="FL Mapped Policy\nArrows represent best action"
# val_max, policy_map = Plots.get_policy_map(pi, V, fl_actions, fl_map_size)
# Plots.plot_policy(val_max, policy_map, fl_map_size, title)

def main():

    gamma = [0.95, 1.00, 1.05]
    epsilon_decay = [0.55, 0.65, 0.75, 1.00]
    iters = [25000, 50000]
    size = generate_random_map(size=8)
    mdp = 'FrozenLake8x8-v1'

    frozenlake = make_env(mdp=mdp, size=size)

    q_learning_results = run_q_gridsearch(process=frozenlake,
                                      gamma=gamma,
                                      epsilon_decay=epsilon_decay,
                                      iters=iters)

    q_learning_results = run_pi_gs(process=frozenlake,
                               gamma=gamma, epsilon_decay=epsilon_decay,
                               iters=iters)

    pi_results = run_pi_gs(process=frozenlake,
                               gamma=gamma, epsilon_decay=epsilon_decay,
                               iters=iters)

    vi_results = run_vi_gs(process=frozenlake,
                               gamma=gamma, epsilon_decay=epsilon_decay, 
                               iters=iters)
    
if __name__ == "__main__":
    main()