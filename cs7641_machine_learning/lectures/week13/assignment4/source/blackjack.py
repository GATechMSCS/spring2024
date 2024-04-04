import gymnasium as gym
from bettermdptools.utils.test_env import TestEnv
from bettermdptools.algorithms.rl import RL
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots
from bettermdptools.utils.blackjack_wrapper import BlackjackWrapper
import numpy as np
import seaborn as sns
from bettermdptools.utils.decorators import add_to
from bettermdptools.utils.grid_search import GridSearch
from bettermdptools.utils.callbacks import MyCallbacks
import matplotlib.pyplot as plt
from gymnasium.spaces import Tuple, Discrete

mdp = 'Blackjack-v1'
size = Tuple([Discrete(32), Discrete(11), Discrete(2)])
gamma=[0.50, 0.75, .99]
iters = [500, 750, 1000]
theta=[.001, .00001]
bl_base = gym.make(id=mdp,
                    render_mode=None)
bl_base.observation_space = size
blackjack = BlackjackWrapper(bl_base)

# FROM UTILITIES
# @add_to(MyCallbacks)
# def on_episode(self, caller, episode):
#     if episode % 1000 == 0:
#     	print(" episode=", episode)

q_learning_results = GridSearch.q_learning_grid_search(blackjack, gamma, iters, theta)
# pi_grid_results = GridSearch.pi_grid_search(blackjack, gamma, iters, theta)
# vi_grid_results = GridSearch.vi_grid_search(blackjack, gamma, iters, theta)

# #test policy
# test_scores = TestEnv.test_env(env=blackjack, n_iters=100, render=False, pi=pi, user_input=False)
# print(np.mean(test_scores))

# FROM PLOTS NOTEBOOK
#run VI
# V, V_track, pi = Planner(blackjack.P).value_iteration()

#create actions dictionary and set map size
# blackjack_actions = {0: "S", 1: "H"}
# blackjack_map_size=(29, 10)

#get formatted state values and policy map
#val_max, policy_map = Plots.get_policy_map(pi, V, blackjack_actions, blackjack_map_size)

#plot policy map
# title="Unedited\nBlackjack Policy Map"
# Plots.plot_policy(val_max, policy_map, blackjack_map_size, title)

# @add_to(Plots)
# @staticmethod
# def modified_plot_policy(val_max, directions, map_size, title):
#     """Plot the policy learned."""
#     sns.heatmap(
#         val_max,
#         annot=directions,
#         fmt="",
#         cmap=sns.color_palette("magma_r", as_cmap=True),
#         linewidths=0.7,
#         linecolor="black",
#     ).set(title=title)
#     img_title = f"Policy_{map_size[0]}x{map_size[1]}.png"
#     plt.show()

# title = "New Blackjack Policy Map"
# Plots.modified_plot_policy(val_max, policy_map, blackjack_map_size, title)