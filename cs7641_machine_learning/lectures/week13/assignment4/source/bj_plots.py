from bettermdptools.utils.plots import Plots
import numpy as np
import seaborn as sns
from bettermdptools.utils.decorators import add_to
from bettermdptools.utils.callbacks import MyCallbacks
import matplotlib.pyplot as plt

np.random.seed(123)

# check
Plots.values_heat_map

# FROM UTILITIES
@add_to(MyCallbacks)
def on_episode(self, caller, episode):
    if episode % 1000 == 0:
    	print(" episode=", episode)

def bj_policy_map(pi, V, blackjack_map_size):
    
    # create actions dictionary and set map size
    blackjack_actions = {0: "S", 1: "H"}
    blackjack_map_size=(29, 10)

    # get formatted state values and policy map
    val_max, policy_map = Plots.get_policy_map(pi, V, blackjack_actions, blackjack_map_size)

    # plot policy map
    title="Blackjack Policy Map"
    Plots.plot_policy(val_max, policy_map, blackjack_map_size, title)

    title = "New Blackjack Policy Map"
    Plots.modified_plot_policy(val_max, policy_map, blackjack_map_size, title)

@add_to(Plots)
@staticmethod
def modified_plot_policy(val_max, directions, map_size, title):
    """Plot the policy learned."""
    sns.heatmap(val_max,
                annot=directions,
                fmt="",
                cmap=sns.color_palette("magma_r", as_cmap=True),
                linewidths=0.7,
                linecolor="black",).set(title=title)
    img_title = f"Policy_{map_size[0]}x{map_size[1]}.png"
    plt.show()