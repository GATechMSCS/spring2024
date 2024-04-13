from bettermdptools.utils.plots import Plots
import numpy as np
import seaborn as sns
from bettermdptools.utils.decorators import add_to
from bettermdptools.utils.callbacks import MyCallbacks
import matplotlib.pyplot as plt

np.random.seed(123)

Plots.values_heat_map
Plots.v_iters_plot
Plots.get_policy_map
Plots.plot_policy

# state values heatmap
def fl_heatmap(size, V):

    Plots.values_heat_map(V, "Frozen Lake\nValue Iteration State Values", size)

def fl_statevalues_iters(V_track):
    
    # state values vs iterations
    # Clip trailing zeros in case convergence is reached before max iterations
    # This is likely when setting the n_iters parameter
    max_value_per_iter = np.trim_zeros(np.mean(V_track, axis=1), 'b')
    Plots.v_iters_plot(max_value_per_iter, "Frozen Lake\nMean Value v Iterations")

def fl_policy_map(pi, V):
    # policy maps
    fl_actions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    fl_map_size=(8,8)
    title="FL Mapped Policy\nArrows represent best action"
    val_max, policy_map = Plots.get_policy_map(pi, V, fl_actions, fl_map_size)
    Plots.plot_policy(val_max, policy_map, fl_map_size, title)


# size=(4,4)
# Plots.values_heat_map(V_vi_gamma, "Frozen Lake\nValue Iteration State Values", size)

# max_value_per_iter = np.trim_zeros(np.mean(v_track_vi_gamma, axis=1), 'b')
# Plots.v_iters_plot(max_value_per_iter, "Frozen Lake\nMean Value v Iterations")

# fl_actions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
# fl_map_size=(4, 4)
# title="FL Mapped Policy\nArrows represent best action"
# val_max, policy_map = Plots.get_policy_map(pi_vi_gamma, V_vi_gamma, fl_actions, fl_map_size)
# Plots.plot_policy(val_max, policy_map, fl_map_size, title)

