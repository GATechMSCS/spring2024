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

def fl_heatmap(size, V):

    # state values heatmap
    Plots.values_heat_map(V, "Frozen Lake\nValue Iteration State Values", size)

def fl_statevalues_iters(V_track):
    
    # state values vs iterations
    max_value_per_iter = np.trim_zeros(np.mean(V_track, axis=1), 'b')
    Plots.v_iters_plot(max_value_per_iter, "Frozen Lake\nMean Value v Iterations")

def fl_policy_map(pi, V, size):
    
    # policy maps
    fl_actions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    title="FL Mapped Policy\nArrows represent best action"
    val_max, policy_map = Plots.get_policy_map(pi, V, fl_actions, size)
    Plots.plot_policy(val_max, policy_map, size, title)