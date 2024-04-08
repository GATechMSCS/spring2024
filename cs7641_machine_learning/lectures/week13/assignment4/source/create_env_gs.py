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

def make_env(mdp, size, slip, render, seed, prob_frozen, ep_steps):

    size = generate_random_map(size=size, p=prob_frozen)

    match mdp:
        case 'FrozenLake8x8-v1':
    
            env = gym.make(id=mdp,
                   desc=size,
                   render_mode=render,
                   is_slippery=slip,
                   max_episode_steps=ep_steps)

        case 'Blackjack-v1':

            env = gym.make(id=mdp,
                       render_mode=render,
                       max_episode_steps=ep_steps)

            env.observation_space = size
            env = BlackjackWrapper(env)
        

    env.reset(seed=seed)
    print(env.render())

    return env

def run_q_gridsearch(process, gamma, epsilon_decay, iters):

    q_learning_results = GridSearch.q_learning_grid_search(env=process, gamma=gamma, epsilon_decay=epsilon_decay, iters=iters)    

    return q_learning_results

def run_pi_gs(process, gamma, theta, iters):
    
    pi_grid_results = GridSearch.pi_grid_search(env=process, gamma=gamma, n_iters=iters, theta=theta)

    return pi_grid_results

def run_vi_gs(process, gamma, theta, iters):

    vi_grid_results = GridSearch.vi_grid_search(env=process, gamma=gamma, n_iters=iters, theta=theta)

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

    gamma_q = [0.90, 0.99, 0.999]
    epsilon_decay_q = [0.90, 0.99, 0.999]
    iters_q = [100_000, 300_000, 500_000]

    gamma_p = [0.90, 0.95, 0.99, 0.999]
    theta_p = [0.01, 0.001, .0001, 0.00001]
    iters_p = [100_000, 400_000, 700_000, 1_000_000]

    gamma_v = [0.90, 0.95, 0.99, 0.999]
    theta_v = [0.01, 0.001, .0001, 0.00001]
    iters_v = [100_000, 400_000, 700_000, 1_000_000]
    
    mdp = 'FrozenLake8x8-v1'
    is_slippery = True
    render_mode = 'ansi'
    prob_frozen = 0.8
    size = 16
    ep_steps = 400

    frozenlake = make_env(mdp=mdp, size=size, slip=is_slippery,
                      render=render_mode,
                      prob_frozen=prob_frozen,
                      seed=seed,
                      ep_steps=ep_steps)

    q_learning_results = run_q_gridsearch(process=frozenlake,
                                      gamma=gamma_q,
                                      epsilon_decay=epsilon_decay_q,
                                      iters=iters_q)

    pi_results = run_pi_gs(process=frozenlake,
                       gamma=gamma_p,
                       theta=theta_p,
                       iters=iters_p)

    vi_results = run_vi_gs(process=frozenlake,
                       gamma=gamma_v, 
                       theta=theta_v,
                       iters=iters_v)
    
if __name__ == "__main__":
    main()