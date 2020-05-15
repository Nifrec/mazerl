"""
This file organizes storing training data to disk,
(any other data than the networks and hyperparameters).
Most importantly it stores and computes he Q-Value overestimation of the agent,
to compare the overestimation of TD3 and DDPG.

Author: Lulof Pirée
"""
# Library imports
import collections
import itertools # For slicing deques.
import numpy as np
import os
from numbers import Number
import torch
# Local imports
from src.agent.network import Network

DEFAULT_PERIOD = 1000
FILENAME_ESTIMATIONS = "values_estimates.txt"
FILENAME_TRAINED_CRITIC_ESTIMATIONS = "values_trained_estimations.txt"
FILENAME_TRUE_VALS = "values_true.txt"
FILENAME_REWARDS = "rewards.txt"
FILENAME_CRITIC_LOSSES = "critic_losses.txt"
FILENAME_ACTOR_LOSSES = "actor_losses.txt"

class Logger():
    """
    Class for logging the overestimation in Q-value prediction, the rewards
    and the actor and critic target losses.
    Useage:
    Use push_rewards() to store a list of all rewards of one episode
    (in chronological order),
    and use push_values() to push value predictions corresponding to the same
    timesteps as the rewards. The value predictions do not need to be pushed
    for an entire episode at once, and can even be pushed one by one.
    Just make sure that the chronological order corresponds!
    Note that push_rewards() automatically also stores the rewards directly
    in a 'rewards' file.
    Use push_critic_losses() and push_actor_losses() to store arrays of
    such losses directly to a file.

    Use update() to (compute and) save the moving average of the 
    true and predicted values to disk.

    NOTE: here 'values' and 'Q-values' and 'sum of iscounted returns' are used
    as synonyms.
    """

    def __init__(self, hyperparameters, period=DEFAULT_PERIOD):
        self.__hyperparameters = hyperparameters
        self.__period = period
        self.__values = collections.deque()
        self.__true_values = collections.deque()

        save_dir = hyperparameters.save_dir
        self.__value_file= open(os.path.join(save_dir, FILENAME_ESTIMATIONS),
                "w+")
        self.__true_value_file= open(os.path.join(save_dir, FILENAME_TRUE_VALS),
                "w+")
        self.__rewards_file = open(os.path.join(save_dir, FILENAME_REWARDS),
                "w+")
        self.__critic_losses_file = \
                open(os.path.join(save_dir, FILENAME_CRITIC_LOSSES),
                "w+")
        self.__actor_losses_file = \
                open(os.path.join(save_dir, FILENAME_ACTOR_LOSSES),
                "w+")
        # Boolean whether this object also logs the estimations of states
        # According to a trained critic.
        self.__has_trained_critic = False

    def has_trained_critic(self):
        return self.__has_trained_critic

    def push_trained_critic_experience(self, state, action):
        if not self.__has_trained_critic:
            raise RuntimeWarning(self.__class__.__name__ 
                + "push_trained_critic_experience(): no critic attached")
        q1, q2 = self.__trained_critic.forward(
                torch.cat((state, action), dim=-1))
        new_estimation = torch.min(q1, q2).item()
        self.__trained_critic_value_file.write(f"{new_estimation}\n")


    def push_rewards(self, episode_rewards):
        """
        Add a list of rewards per timestep of a single episode to the
        logger. Used to compute the moving average of the 'true values'.
        Also appends rewards directly to rewards file.

        Arguments:
        * episode_rewards: list, rewards per timestep of an episode.
        """
        assert(isinstance(episode_rewards, list)), \
            "push_rewards: list expected"
        self.__store_rewards(episode_rewards)
        new_true_values = self.__compute_true_values(episode_rewards)
        self.__true_values += new_true_values

    def push_value_estimations(self, value_estimations):
        """
        Adds a list of new value estimates (or a single number), used
        to compute more moving average values of the estimated values.

        Arguments:
        * value_estimations: list, Q-value predictions in chronological 
            order of episodes and timesteps.
            Alternatively, a single number is also accepted.
        """
        if (isinstance(value_estimations, list)):
            self.__values += value_estimations
        elif (isinstance(value_estimations, Number)):
            self.__values.append(value_estimations)
        else:
            raise ValueError("pushing illegal type"
                + f"{type(value_estimations)} for value_estimations")

    def __compute_true_values(self, episode_rewards):
        """
        Takes a list of rewards per timestep of an episode,
        and uses the definition of the Q-values 
        (i.e. brute-force computing sum of discounted rewards)
        to compute the 'true' discounted future returns for each timestep.

        Arguments:
        * episode_rewards: list, rewards per timestep of an episode.
        """
        episode_true_values = []
        discounted_return = 0
        for reward in reversed(episode_rewards):
            discounted_return = reward \
                    + self.__hyperparameters.discount_rate * discounted_return
            episode_true_values.append(discounted_return)
        episode_true_values.reverse()
        return episode_true_values

    def can_update(self):
        """
        Checks if currently enough new data has been collected (via pushes)
        to compute new moving averages and to store them.

        Returns:
        * boolean, whether there are sufficient new datapoints to update.
        """
        if (len(self.__true_values) < self.__period) \
            or (len(self.__values) < self.__period):
            return False
        else:
            return True

    def update(self):
        """
        Computes new moving average values of true and estimated Q-values
        and appends them to the files on disk.
        Continues as long as there is enough data to compute the 
        moving averages.
        """
        print("Updating logger.")
        num_saved_values = 0
        while(self.can_update()):
            last_period_values = list(itertools.islice(self.__values, 0,
                self.__period))
            new_estimate_average = np.mean(last_period_values)
            self.__values.popleft()
            self.__store_value(new_estimate_average)

            last_period_true_values = list(itertools.islice(self.__true_values, 0, 
                    self.__period))
            new_true_average = np.mean(last_period_true_values)
            self.__true_values.popleft()
            self.__store_true_value(new_true_average)

            num_saved_values += 1

        print(f"Logger stored {num_saved_values} new overestimation values.")

    def __store_value(self, new_value):
        self.__value_file.write(f"{new_value}\n")

    def __store_true_value(self, new_true_value):
        self.__true_value_file.write(f"{new_true_value}\n")

    def close_files(self):
        self.__value_file.close()
        self.__true_value_file.close()
        self.__rewards_file.close()
        self.__critic_losses_file.close()
        self.__actor_losses_file.close()
        if self.__has_trained_critic:
            self.__trained_critic_value_file.close

    def push_critic_loss(self, critic_loss):
        """
        Appends new loss to the critic_losses file.
        """
        self.__critic_losses_file.write(str(critic_loss) + "\n")

    def push_actor_loss(self, actor_loss):
        """
        Appends new loss to actor_losses file.
        """
        self.__actor_losses_file.write(str(actor_loss) + "\n")

    def __store_rewards(self, rewards):
        """
        Appends new rewards to text file of processed rewards.
        """
        for reward in rewards:
            self.__rewards_file.write(str(reward)+"\n")

    def attach_trained_critic(self, critic):
        """
        Add a trained critic, such that when logging the overestimations,
        also the estimations according to this trained critic for the given
        state-action pairs are stored in a separate file.
        """
        assert(isinstance(critic, Network))
        self.__trained_critic = critic
        self.__trained_critic_value_file = \
                open(
                    os.path.join(
                        self.__hyperparameters.save_dir,
                        FILENAME_TRAINED_CRITIC_ESTIMATIONS
                    ),
                    "w+"
                )
        self.__has_trained_critic = True