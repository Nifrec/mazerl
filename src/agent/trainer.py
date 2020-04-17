"""
Class that implements the main iteration of a DDPG/TD3 training loop.

Author: Lulof Pirée

Possible future enhancements:
* Decay the noise on action selection over the course of training.
"""

import gym
import math
import random
import numbers
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
from .agent import Agent
from .td3_agent import TD3Agent

class Trainer:
    """
    Class for training, saving and executing a DDPG or TD3 agent.
    """

    def __init__(self, hyperparameters):
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
        self.hyperparameters = hyperparameters
        self.env = hyperparameters.env
        self.mode = hyperparameters.mode
        self.state_size = len(self.env.reset())
        self.logger = None
        self.training_is_set_up = False

        # Prefer CUDA (=GPU), which is normally faster.
        self.device = \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__create_agent()
        
    def __create_agent(self):
        """
        Sets up agent that trains according to DDPG ot TD3
        depending on self.mode.
        """
        if (self.mode == "TD3"):
            self.agent = TD3Agent(self.hyperparameters, self.device,
                    self.state_size, self.hyperparameters.save_dir)
        elif (self.mode == "DDPG"):
            self.agent = Agent(self.hyperparameters, self.device, 
                    self.state_size, self.hyperparameters.save_dir)
        else:
            raise ValueError(self.__class__.__name__ + "invalid mode")

    def train(self):
        if not self.training_is_set_up:
            raise RuntimeError(self.__class__.__name__ + ".train(): please "
                    + "set up training with " + self.__class__.__name__ 
                    + ".setup_train_run() first" )

        replay_memory = ReplayMemory(self.hyperparameters.memory_capacity)
        rewards_per_episode = [] # For plot
        self.critic_losses = []
        self.actor_losses = []

        for episode in range(1, self.hyperparameters.num_episodes+1):
            state = self.env.reset()
            episode_rewards = []
            # The neural networks only accept tensors as input.
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            
            for timestep in range(self.hyperparameters.max_episode_duration):
                action = self.__choose_action(state, episode)

                next_state, reward, done, _ = self.env.step(
                        action.tolist()[0])
                episode_rewards += [reward]
                # Store experience as 4 tensors, since used in neural networks.
                next_state = torch.tensor([next_state],
                        dtype=torch.float).to(self.device)
                reward = torch.tensor([reward], dtype=torch.float)\
                        .to(self.device)
                done = torch.tensor([done], dtype=torch.bool)\
                        .to(self.device)
                new_experience = Experience(state, action, reward, next_state,
                        done)
                replay_memory.push(new_experience)

                if (replay_memory.can_sample(self.hyperparameters.batch_size)):
                    batch = replay_memory.sample(
                            self.hyperparameters.batch_size)
                    self.update_agent(episode, batch)

                if (done.item()):
                    print(f"Agent reached goal in episode {episode}",
                            f"at timestep {timestep}")
                    break
                    
                state = next_state
                    
            self.__process_training_data(episode, rewards_per_episode,
                    episode_rewards)

        self.logger.close_files()
        # Otherwise may end up overwriting previous train run-data
        self.training_is_set_up = False

    def __choose_action(self, state, episode):
        """
        Lets agent choose either a random action at the beginning
        or an action according to its inference actor.
        """
        select_random = (episode < self.hyperparameters.random_action_episodes)
        action, value_estimation = self.agent.choose_action(state,
                select_random)

        if (self.logger is not None):
            self.logger.push_value_estimations(value_estimation)
            if (self.logger.has_trained_critic()):
                self.logger.push_trained_critic_experience(
                    state.to(self.device), action)

        return action

    def update_agent(self, episode, batch):
        critic_loss, actor_loss = self.agent.update(batch, episode)
        if self.logger is not None:
            self.logger.push_critic_loss(critic_loss)
            if actor_loss is not None:
                self.logger.push_actor_loss(actor_loss)

    def sample_batch(self, replay_memory):     
        return replay_memory.sample(self.hyperparameters.batch_size)

    def __process_training_data(self, episode, rewards_per_episode,
            episode_rewards):
        rewards_per_episode.append(sum(episode_rewards))
            
        if (self.logger is not None):
            self.logger.push_rewards(episode_rewards)

        if (episode % self.hyperparameters.plot_interval == 0):
            self.__plot_and_print(episode, rewards_per_episode)

        if (episode % self.hyperparameters.checkpoint_interval == 0):
            self.__save_progress(episode, rewards_per_episode)
            self.logger.update()

    def __plot_and_print(self, episode, rewards_per_episode):
        plot_reward_and_moving_average(
                        self.hyperparameters.moving_average_period,
                        rewards_per_episode)
        self.__print_train_progress(episode, rewards_per_episode)

    def __print_train_progress(self, episode, rewards_per_episode):
        period = self.hyperparameters.plot_interval//10
        print(f"Finished episode {episode}")
        print(f"Rewards last {period} episodes:", rewards_per_episode[-period:])

    def __save_progress(self, episode, rewards_per_episode):
        self.agent.save_checkpoint()
        plot_reward_and_moving_average(
                self.hyperparameters.moving_average_period,
                rewards_per_episode, 
                self.hyperparameters.save_dir
        )

    def load_checkpoint(self, checkpoint_dir = None):
        if (checkpoint_dir == None):
            checkpoint_dir = self.hyperparameters.save_dir
        
        self.agent.load_checkpoint(checkpoint_dir)

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
        state = self.env.reset()
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        
        for timestep in range(max_episode_duration):
            self.env.render()
            action, _ = self.agent.choose_action(state,
                    self.hyperparameters.exploration_noise_std)
            next_state, _, done, _ = self.env.step(action.tolist()[0])

            if (done):
                return
            else:
                next_state = torch.tensor([next_state], 
                        dtype=torch.float).to(self.device)
                state = next_state
        # Also render final state after last action
        self.env.render()
        print("Demo completed")

    def setup_train_run(self):
        """
        Attaches a logger to log training data and creates
        a directory to save generated data 
        (parameters and performance stats) to.
        """
        # Make directory to store all data and log the hyperparameters.
        print(f"Initialising {self.hyperparameters.mode} with"
                + f"self.hyperparameters: {self.hyperparameters}")
        setup_save_dir(self.hyperparameters.save_dir)
        make_setup_info_file(self.hyperparameters)\
        
        self.training_is_set_up = True

    def attach_logger(self, logger):
        assert isinstance(logger, Logger)

        self.logger = logger