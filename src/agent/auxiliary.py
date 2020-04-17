"""
Helper functions of the DDPG/TD3 implementation.

Author: Lulof Pirée (Nifrec).
"""
import matplotlib.pyplot as plt
import torch
import time
import os
import random
import collections

# This is a class not an instance!
# See main.py for example use and more explanation.
HyperparameterTuple = collections.namedtuple("HyperparameterTuple", 
                                 ["env", "mode", "device",
                                  "discount_rate", "learning_rate_critic",
                                  "learning_rate_actor",
                                  "exploration_noise_std",
                                  "min_action", "max_action",
                                  "num_episodes", "batch_size", 
                                  "max_episode_duration",
                                  "memory_capacity", "polyak",
                                  "plot_interval", "moving_average_period", 
                                  "checkpoint_interval", "action_size",
                                  "save_dir", "random_action_episodes",
                                  "td3_critic_train_noise_std",
                                  "td3_critic_train_noise_bound",
                                  "td3_target_and_actor_update_interval"])

# This is a class not an instance!
Experience = collections.namedtuple("Experience", 
        ["state", "action", "reward", "next_state", "done"])

def compute_moving_average(values, period):
    """
    Checks if possible to compute the moving average, if
    possible computes moving average over period.
    If not possible returns None.

    Arguments:
    * values: array
        Array of numbers to compute the moving average of.
    * period: int
        Amount of values to consider when computing
        the running average (which is simply the average 
        over the last [period] values).

    Returns:
    * moving_avg: numpy.array
        Same length as values, average value of values
        at index & previous (period - 1) values. For values
        with an index < (period - 2), None will be returned.
    """        
    if (len(values) >= period):
        return compute_moving_average_when_enough_values(values, period)
    else:
        return None
        
def compute_moving_average_when_enough_values(values, period):
    """
    Computes current the moving average values of an
    array of values. Assumes len(values) < period.

    Arguments:
    * values: array
        Array of numbers to compute the moving average of.
    * period: int
        Amount of values to consider when computing
        the running average (which is simply the average 
        over the last [period] values).

    Returns:
    * moving_avg: numpy.array
        Same length as values, average value of values
        at index & previous (period - 1) values. For values
        with an index < (period - 2) the input values and not
        the average will be used.
    """
    assert (len(values) >= period)

    moving_avg = torch.tensor(values, dtype=torch.float)
    # unfold starts at index 0, adds a slice
    # [index:index+size] to the output, increases the
    # index by [step] and repeats.
    moving_avg = moving_avg.unfold(dimension=0, size=period,
                            step=1)
    # Dim=1 because need the average of each slice.
    moving_avg = moving_avg.mean(dim=1).flatten(start_dim=0)
    # Cannot compute moving average for [period-1] first 
    # episodes, because need at least [period] items.
    # Fill missing values with 0's.
    moving_avg = torch.cat((torch.tensor(values[0:period]), moving_avg))

    return moving_avg.numpy()

def plot_reward_and_moving_average(moving_average_period, rewards,
        save_directory_name=None):
    """
    Plot the rewards the agent earned and the running average in a single
    pyplot figure.
    
    Arguments:
        * moving_average_period: int
            Amount of values to consider when computing
            the running average (which is simply the average 
            over the last [moving_average_period] values).
        * rewards: array of numbers
            Values to plot and to compute the moving averages over.
    """
    if not isinstance(moving_average_period, int):
            raise ValueError("RewardValueAndRunningAveragePlotter:"                             
                    + "illegal argument for moving_average_period")
    # Used to ensure the same pyplot figure is updated every time
    FIGNUMBER=1
    plt.figure(FIGNUMBER)
    plt.cla()
    plt.plot(rewards, color="blue")
    moving_av = compute_moving_average(
        rewards, moving_average_period)
    if moving_av is not None:
        plt.plot(moving_av, color="red")

    if (save_directory_name != None):
        plt.savefig(os.path.join(save_directory_name, "train_plot.svg"))
    plt.pause(0.001)

def get_timestamp():
    """
    Returns formatted string of current time,
    as DD_MM_YYYY-HH_MM (Day Month Year Hours Minutes)
    """
    time_struct = time.localtime(time.time())
    string = "{:02d}-{:02d}-{}-{:02d}_{:02d}".format(
            time_struct[2], time_struct[1],
            time_struct[0], time_struct[3],
            time_struct[4])
    return string

def setup_save_dir(directory_name):
    try:
        # If the program ran less than a minute ago this will fail.
        os.mkdir(directory_name)
    except:
        pass

def make_setup_info_file(hyperparameters):
    f = open(os.path.join(hyperparameters.save_dir, "setup_info.txt"), "w+")
    f.write(hyperparameters.env_name + "\n")
    f.write(str(hyperparameters))
    f.close()

def create_noise(noise_scale):
    """
    Create a random number from the uniform distribution
    [-1, 1] multiplied by noise_scale.
    """
    return noise_scale * random.uniform(-1, 1)

def clip(value, minimum, maximum):
    """
    Ensures the output is clipped into [minimum, maximum].
    Simply returns input value if this is already the case.
    
    WARNING: will convert a tensor to a normal python primitive.
    NOTE: also available as toch.clamp(...). This will return a tensor.
    """
    return max(min(value, maximum), minimum)