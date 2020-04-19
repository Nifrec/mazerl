"""
Class that implements the main iteration of a DDPG/TD3 training loop.

Author: Lulof Pirée

Possible future enhancements:
* Decay the noise on action selection over the course of training.
"""

import gym
import math
import random
from numbers import Number
from typing import Iterable, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import collections
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .network import Network
from .replay_memory import ReplayMemory
from .auxiliary import Experience, HyperparameterTuple
from .auxiliary import compute_moving_average, \
        compute_moving_average_when_enough_values, \
        plot_reward_and_moving_average, clip,\
        setup_save_dir, make_setup_info_file
from .logger import Logger
from .agent_class import Agent
from .td3_agent import TD3Agent


class Trainer:
    """
    Class for training, saving and executing a DDPG or TD3 agent.
    """

    def __init__(self, hyperparameters: HyperparameterTuple,
            agent: Agent, logger: Logger):
        """
        Initialize the DDPG training algorithm, set up Gym environment,
        create file with setup info.
        Can run in either DDPG or TD3 mode, which vary slightly in optimization
        and learning algorithm used.
        
        Arguments:
        * hyperparameters: HyperparameterTuple, with training setup.
        * env_name: string, name of environment known by gym.
        * mode: string, "DDPG" or "TD3"
        """
        self.__hyperparameters = hyperparameters
        self.__env = hyperparameters.env
        self.__agent = agent
        self.__logger = logger

        # Prefer CUDA (=GPU), which is normally faster.
        self.__device = \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):

        replay_memory = ReplayMemory(self.__hyperparameters.memory_capacity)
        rewards_per_episode = [] # For plot
        self.critic_losses = []
        self.actor_losses = []

        for episode in range(1, self.__hyperparameters.num_episodes+1):
            state = self.__env.reset()
            episode_rewards = []
            # The neural networks only accept tensors as input.
            state = torch.tensor([state], dtype=torch.float).to(self.__device)
            
            for timestep in range(self.__hyperparameters.max_episode_duration):
                action = self.__choose_action(state, episode)
                print(action)
                next_state, reward, done, _ = self.__env.step(
                        action[0].tolist())
                episode_rewards += [reward]
                # Store experience as 4 tensors, since used in neural networks.
                next_state = torch.tensor([next_state],
                        dtype=torch.float).to(self.__device)
                reward = torch.tensor([reward], dtype=torch.float)\
                        .to(self.__device)
                done = torch.tensor([done], dtype=torch.bool)\
                        .to(self.__device)
                new_experience = Experience(state, action, reward, next_state,
                        done)
                replay_memory.push(new_experience)

                if (replay_memory.can_sample(self.__hyperparameters.batch_size)):
                    batch = replay_memory.sample(
                            self.__hyperparameters.batch_size)
                    self.__update_agent(episode, batch)

                if (done.item()):
                    print(f"Agent reached goal in episode {episode}",
                            f"at timestep {timestep}")
                    break
                    
                state = next_state
                    
            self.__process_training_data(episode, rewards_per_episode,
                    episode_rewards)

        self.__logger.close_files()

    def __choose_action(self, state: torch.Tensor, episode: int) \
            -> Iterable[Number]:
        """
        Lets agent choose either a random action at the beginning
        or an action according to its inference actor.
        """
        select_random = \
                (episode < self.__hyperparameters.random_action_episodes)
        action, value_estimation = self.__agent.choose_action(state,
                select_random)

        if (self.__logger is not None):
            self.__logger.push_value_estimations(value_estimation)
            if (self.__logger.has_trained_critic()):
                self.__logger.push_trained_critic_experience(
                        state.to(self.__device), action)
        return action

    def __update_agent(self, episode: int, batch: Tuple[Experience]):
        critic_loss, actor_loss = self.__agent.update(batch, episode)
        if self.__logger is not None:
            self.__logger.push_critic_loss(critic_loss)
            if actor_loss is not None:
                self.__logger.push_actor_loss(actor_loss)

    def __process_training_data(self, episode, rewards_per_episode,
            episode_rewards):
        rewards_per_episode.append(sum(episode_rewards))
            
        if (self.__logger is not None):
            self.__logger.push_rewards(episode_rewards)

        if (episode % self.__hyperparameters.plot_interval == 0):
            self.__plot_and_print(episode, rewards_per_episode)

        if (episode % self.__hyperparameters.checkpoint_interval == 0):
            self.__save_progress(episode, rewards_per_episode)
            self.__logger.update()

    def __plot_and_print(self, episode, rewards_per_episode):
        plot_reward_and_moving_average(
                        self.__hyperparameters.moving_average_period,
                        rewards_per_episode)
        self.__print_train_progress(episode, rewards_per_episode)

    def __print_train_progress(self, episode, rewards_per_episode):
        period = self.__hyperparameters.plot_interval//10
        print(f"Finished episode {episode}")
        print(f"Rewards last {period} episodes:", rewards_per_episode[-period:])

    def __save_progress(self, episode, rewards_per_episode):
        self.__agent.save_checkpoint()
        plot_reward_and_moving_average(
                self.__hyperparameters.moving_average_period,
                rewards_per_episode, 
                self.__hyperparameters.save_dir
        )

    def load_checkpoint(self, checkpoint_dir=None):
        if (checkpoint_dir == None):
            checkpoint_dir = self.__hyperparameters.save_dir
        
        self.__agent.load_checkpoint(checkpoint_dir)

    def show_demo(self, max_episode_duration):
        """
        Lets the agent play one episode while the enviroment is rendered.
        No random noise will be added to the agent's actions, nor will
        any training occur.
        This is just to show how it behaves.
        Arguments:
        * max_episode_duration: int, maximum amount of timesteps before
                episode will be stopped.
        """
        state = self.__env.reset()
        state = torch.tensor([state], dtype=torch.float).to(self.__device)

        for timestep in range(max_episode_duration):
            self.__env.render()
            action, _ = self.__agent.choose_action(state,
                    self.__hyperparameters.exploration_noise_std)
            next_state, _, done, _ = self.__env.step(action.tolist()[0])

            if (done):
                return
            else:
                next_state = torch.tensor([next_state], 
                        dtype=torch.float).to(self.__device)
                state = next_state
        # Also render final state after last action
        self.__env.render()
        print("Demo completed")