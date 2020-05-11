"""
Main file to organize Reinforcement Learning using TD3 and Asynchronous
Methods. The responsibility of the AsynchronousTrainer is to organize the
agent, networks and experiences, and to take the timesteps of the
environment.


Author: Lulof Pirée
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
import multiprocessing
import threading
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .network import Network
from .replay_memory import ReplayMemory
from agent.auxiliary import Experience, AsyncHyperparameterTuple
from .auxiliary import compute_moving_average, \
        compute_moving_average_when_enough_values, \
        plot_reward_and_moving_average, clip,\
        setup_save_dir, make_setup_info_file, \
        create_environment
        
from .logger import Logger
from .agent_class import Agent
from .td3_agent import TD3Agent



class AsynchronousTrainer:
    """
    Alternative to Trainer: uses multiple agents asyncronously
    to update a shared network, does not use ReplayMemory.
    This saves a lot of video-memory.

    Does *not* support logging of pre-trained-critic-estimations via the 
        provided logger.
    """

    def __init__(self, hyperparameters: AsyncHyperparameterTuple,
            agents: Tuple[Agent, ...], logger: Logger, num_processes: int):
        """
        Initialize the TD3 training algorithm, set up Gym environment,
        create file with setup info.
        
        Arguments:
        * hyperparameters: HyperparameterTuple, with training setup.
        * agent: agent to hold the actor and the critic and to choose moves.
        * logger: instance to store meta-training data to a file.
        * num_processes: how many processes should be launched with the provided
            agents. If higher than the number of agents will wrap around
            the agent list.
        """
        self.__hyperparameters = hyperparameters
        self.__agents = agents
        self.__logger = logger
        self.__num_processes = num_processes
        # Prefer CUDA (=GPU), which is normally faster.
        device = \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):

        gradientQueue = multiprocessing.Queue()
        processes = []

        num_agents = len(self.__agents)
        i = 0
        for _ in range(self.__num_processes):
            agent = self.__agents[i]
            i = (i + 1) % num_agents

            worker = AgentWorker(self.__hyperparameters, agent, gradientQueue)
            #p = multiprocessing.Process(target=worker.run)
            p = threading.Thread(target=worker.run)
            processes.append(p)
            p.start()

    def process_gradient_queue(self):
        """
        Enter a loop in which a gradientQueue is constantly being checked
        for new gradients. When any occur they are used to
        update the actor and critic networks.
        """
        pass #TODO

class AgentWorker:
    """
    Represents a single agent process. Responsible for letting an agent
    progress timesteps in an evironment, and collects gradients with
    respect to the weights of ther actor- and critic-networks.
    """

    def __init__(self, hyperparameters: AsyncHyperparameterTuple,
        agent: Agent, gradientQueue: multiprocessing.Queue):
        """
        Arguments:
        * hyperparameters: HyperparameterTuple, with training setup.
        * agent: agent to hold the actor and the critic and to choose moves.
        * gradientQueue: multiprocessing queue used to collect tuples
            of (actor_grad, critic_grad), i.e. optimization changes
            to add to parameters of networks.
        """
        self.__hyperparameters = hyperparameters
        self.__env = create_environment(hyperparameters.env_enum_name)
        self.__agent = agent
        self.__queue = gradientQueue

        # Shorthand for often used hyperparameter.
        self.__device = self.__hyperparameters.device

    def run(self):
        
        for episode in range(1, self.__hyperparameters.num_episodes+1):
            print(f"Worker (id:{id(self)}, pid:{os.getpid()}) " \
                    + f"starts ep:{episode}")
            
            state = self.__env.reset()
            episode_rewards = []
            # The neural networks only accept tensors on the same device.
            # Will be copied to a new tensor for the optimization,
            # so can safely disable gradient tracking here.
            state = torch.tensor([state], dtype=torch.float, 
                    requires_grad=False).to(self.__device)
            
            for timestep in range(self.__hyperparameters.max_episode_duration):
                reward, next_state, done = self.__make_timestep(state, episode)
                episode_rewards.append(reward.item())

                if done.item():
                    break

                # Copy the tensor value in-place to the GPU.
                state.copy_(next_state)
            
    def __make_timestep(self, state, episode) -> Tuple[int, torch.Tensor, bool]:
        action = self.__choose_action(state, episode)
        next_state, reward, done, _ = self.__env.step(
                action[0].tolist())
        # Store experience as 4 tensors, since used in neural networks.
        next_state = torch.tensor([next_state],
                dtype=torch.float).to(self.__device)
        reward = torch.tensor([reward], dtype=torch.int,
                requires_grad=False).to(self.__device)
        done = torch.tensor([done], dtype=torch.bool, 
                requires_grad=False).to(self.__device)
        experience = Experience(state, action, reward, next_state,
                done)

        self.__accumulate_grads_from_experience(experience)

        return reward, state, done

    def __accumulate_grads_from_experience(self, experience: Experience):
        batch = [experience]
        self.__agent.update_critic_net(batch, gradient_only=True)
        self.__agent.update_actor_net(batch, gradient_only=True)
    
    def __collect_gradients(self):
        """
        Collects the gradients from the networks and sends them to the
        gradient-queue for another process/thread to use them.
        """
        grads = self.__agent.get_gradients()
        self.__queue.put(grads)
        self.__agent.reset_gradients()


    def __choose_action(self, state: torch.Tensor, episode: int) \
            -> Iterable[Number]:
        """
        Lets agent choose either a random action at the beginning
        or an action according to its inference actor.
        """
        select_random = \
                (episode < self.__hyperparameters.random_action_episodes)
        action, _ = self.__agent.choose_action(state, select_random)
        return action
 