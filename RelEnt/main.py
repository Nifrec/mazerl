from Grid import Grid
from PolicyIterator import PolicyIterator
from RelEnt import RelEnt

import matplotlib.pyplot as plt

""" http://proceedings.mlr.press/v15/boularias11a/boularias11a.pdf """
# NOTE:: using "coord", "cell" and sometimes "point" words, but they mean the same thing
# NOTE:: x and y are kinda flipped throughout the code
"""
States locations as in the resulting pcolor graphs (with grid size 3):
    6 7 8
    3 4 5
    0 1 2
"""


def perform_descrete_RelEnt(grid_size, n_demos):
    gw = Grid(grid_size)
    init_rewards = gw.rewards

    # Obtain the optimal policy for the environment to generate optimal demonstrations
    pi = PolicyIterator(gw)
    optimal_policy = pi.policy_iteration(100)

    # Generate simulated demos
    optimal_demos = gw.generate_demos(optimal_policy, n_demos)
    nonoptimal_demos = gw.generate_demos(None, n_demos)

    # Train RelEnt
    relent = RelEnt(optimal_demos, nonoptimal_demos)
    relent.train()

    trained_rewards = relent.weights.reshape(grid_size, grid_size)

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    c1 = ax[0].pcolor(init_rewards, edgecolors='k', linewidths=2)
    plt.colorbar(c1, ax=ax[0])
    ax[0].set_title("Initial reward")

    c2 = ax[1].pcolor(trained_rewards, edgecolors='k', linewidths=2)
    plt.colorbar(c2, ax=ax[1])
    ax[1].set_title("Recovered reward")

    fig.show()
    print(" ✓")


perform_descrete_RelEnt(30, 10)