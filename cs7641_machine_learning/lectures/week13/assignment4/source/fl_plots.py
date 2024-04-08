from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots
import numpy as np
import seaborn as sns
from bettermdptools.utils.decorators import add_to
from bettermdptools.utils.callbacks import MyCallbacks
import matplotlib.pyplot as plt

np.random.seed(123)

# FROM PLOTS NOTEBOOK
# state values heatmap
frozen_lake = gym.make('FrozenLake8x8-v1', render_mode=None)
V, V_track, pi = Planner(frozen_lake.P).value_iteration(n_iters=5000)
size=(8,8)
Plots.values_heat_map(V, "Frozen Lake\nValue Iteration State Values", size)

# state values vs iterations
# Clip trailing zeros in case convergence is reached before max iterations
# This is likely when setting the n_iters parameter
max_value_per_iter = np.trim_zeros(np.mean(V_track, axis=1), 'b')
Plots.v_iters_plot(max_value_per_iter, "Frozen Lake\nMean Value v Iterations")

# policy maps
fl_actions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
fl_map_size=(8,8)
title="FL Mapped Policy\nArrows represent best action"
val_max, policy_map = Plots.get_policy_map(pi, V, fl_actions, fl_map_size)
Plots.plot_policy(val_max, policy_map, fl_map_size, title)
