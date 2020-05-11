"""
Class representing the default learning agent for the TD3/DDPG training.
The Agent is responsible for choosing moves given a state, and uses two
neural networks (an actor and a critic) for this.
While functional by itself, also serves as a base class for
Trainer-compatible actors.

Author: Lulof Pirée
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
from typing import Tuple, Iterable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .auxiliary import Experience, HyperparameterTuple
from .critic_network import CriticNetwork
from .actor_network import ActorNetwork
from .logger import Logger
from .auxiliary import Mode
from .replay_memory import ReplayMemory


class Agent():
    """
    Class that keeps track of/does the following:
    * Store the neural networks.
    * Train the neural networks.
    * Select and clip actions.
    """

    def __init__(self, hyperparameters: HyperparameterTuple, 
            actor_net: ActorNetwork, critic_net: CriticNetwork):
        """
        Initialize the agent.
        Parameters:
        * hyperparameters: Hyperparameters object (subclass of named_tuple)
        * device: string, device supported by PyTorch (usually "GPU" or "CPU")
        * state_size: int, length of expected state vector 
                (depends on environment)
        * checkpoint_dir: string, name of directory to save checkpoints to.
        """
        
        self.hyperparameters = hyperparameters
        self.actor = actor_net
        self.critic = critic_net
        self.__device = hyperparameters.device
        self.__create_target_networks()
        self.__set_checkpoint_dir(hyperparameters.save_dir)
        

    def __create_target_networks(self):        
        self.actor_target = type(self.actor)(self.actor.input_size,
                self.actor.output_size)
        self.critic_target = type(self.critic)(self.critic.input_size,
                self.critic.output_size, self.critic.mode)
        
        # Clone also random initialization to target networks.
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.to(self.__device)
        self.critic_target.to(self.__device)
        # Targets never get trained, weights are always directly copied instead.
        self.actor_target.eval()
        self.critic_target.eval() 

        self.critic_optim = optim.Adam(params=self.critic.parameters(),
                lr=self.hyperparameters.learning_rate_critic)
        self.actor_optim = optim.Adam(params=self.actor.parameters(), 
                lr=self.hyperparameters.learning_rate_actor)
        
    def choose_action(self, state: torch.Tensor, select_random: bool=False) \
            -> Tuple[torch.Tensor, float]:
        """
        Query the actor-network and return the action chosen.
        Optionally random (normally )distributed noise in [-1, 1] can be added,
        which will be drawn from a random distribution with std=[noise_std].
        NOTE: for noise_std = 0, this will result in simply no noise added.
                (noise_std is a hyperparameter)
        NOTE: will store the predicted Q-value to disk if logger attached.

        Arguments:
        * state: torch.tensor of size [self.state_size].
        * select_random: returns random value(s) in [-1, 1]

        Returns:
        * action: torch.tensor of shape self.hyperparameters.action_size,
                stored on self.__device.
        * estimation: float, estimated Q-value for this state-action pair.
        """
        if (select_random):
            action = (torch.rand(self.hyperparameters.action_size)-0.5)*2
            # Need to be on same device as network for value_estimation.
            action = action.unsqueeze(0).to(self.__device)
        else:
            noise_std = self.hyperparameters.exploration_noise_std
            action = self.actor(state) + np.random.normal(0, noise_std)
            action += torch.empty_like(action).normal_(mean=0, std=noise_std)
            action = self.__clamp_action(action)

        value_estimation = self.get_value_estimation(state, action)
        return action, value_estimation

    def __clamp_action(self, action):
        return torch.clamp(
                action,
                self.hyperparameters.min_action, 
                self.hyperparameters.max_action
        )
        
    def get_value_estimation(self, state, action):
        """
        Get estimated Q-value according to learned Q-function
        for state-action pair.

        Arguments:
        * state: torch.tensor of shape self.state_size.
        * action: torch.tensor of shape self.hyperparameters.action_size.

        Returns:
        * float: estimated Q-value.
        """
        return self.critic.forward(state, action).item()

    def update(self, batch: Tuple, episode: int):
        """
        Takes a batch of Experience instances and uses them to update the
        inference networks using gradient ascent/descent.
        Arguments:
        * batch: iterable of Experience instances.
        * episode: unused, only here to provide same interface as
                TD3Agent.
        

        Returns:
        * critic_loss, actor_loss: floats, losses (i.e. values used
                to optimize the networks) for critic and actor respectively.
        """
        self.reset_gradients()
        critic_loss = self.update_critic_net(batch).item()
        actor_loss = self.update_actor_net(batch).item()
        self.update_target_networks()
        return critic_loss, actor_loss
                
    def get_gradients(self) -> Tuple[list, list]:
        """
        Returns a list of the the gradient part of the parameters,
        for both networks.
        The gradients are in order of the parameters as returned by
        nn.Module.named_parameters()
        """
        # https://discuss.pytorch.org/t/please-help-how-can-copy-the-gradient-from-net-a-to-net-b/41226/8
        #for net1,net2 in zip(A.named_parameters(),B.named_parameters()):
        #        net2[1].grad = net1[1].grad.clone()
        critic_grad = [
                param[1].grad.clone for param in self.critic.named_parameters()
        ]
        actor_grad = [
                param[1].grad.clone for param in self.actor.named_parameters()
        ]
        return critic_grad, actor_grad

    def reset_gradients(self):
        """
        Sets the accumulated gradients in the actor and critic networks to 0.
        """
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

    def update_critic_net(self, batch: Iterable, gradient_only: bool=False):
        """
        * gradient_only: bool, if flagged True will compute the gradient but
                not modify network weights. 
        NOTE: Will not erase previous gradients (so they accumulate instead)
        """
        states, actions, rewards, next_states, dones \
            = ReplayMemory.extract_experiences(batch)
        self.critic.train()
        self.actor.eval()
        
        # Current Q-Value prediction based on historical state-action pair
        # = 'bad' thing to improve
        # These can be different than at time of gathering the experience.
        # This does not matter and is a feature not a bug!
        values = self.critic.forward(states, actions)
        # Compute Q-Value of next state + action (from target actor) according
        # to target critic. These are the 'good' examples.
        target_actions = self.compute_target_action(next_states)
        target_values = self.critic_target.forward(next_states, target_actions)
        targets = rewards.unsqueeze(dim=-1) \
                + (dones == False) \
                    * self.hyperparameters.discount_rate * target_values
        
        # Perform backprop
        loss = ((targets - values)**2).mean()
        loss.backward()
        if not gradient_only:
            self.critic_optim.step()
        self.critic.eval()
        return loss

    def compute_target_action(self, next_states):
        return self.actor_target.forward(next_states)

    def update_actor_net(self, batch: Iterable, gradient_only: bool=False):
        """
        Perform forward prop on actor, with as loss the predicted value
        of the predicted action in that state (not from target but from normal
        networks).
        * gradient_only: bool, if flagged True will compute the gradient but
                not modify network weights.
        NOTE: Will not erase previous gradients (so they accumulate instead).
        """
        states, _, _, _, _ = ReplayMemory.extract_experiences(batch)
        self.critic.eval()
        actions = self.actor.forward(states)
        values = self.critic.forward(states, actions)
        self.actor.train()
        loss = ((-1*values).mean())
        loss.backward()
        if not gradient_only:
            self.actor_optim.step()
        self.actor.eval()

        return loss

    def update_target_networks(self):
        """
        Update target networks using polyak averaging
        (i.e. copy weights of inference networks parially)
        """
        self.__update_target_network_polyak(self.hyperparameters.polyak,
                self.actor, self.actor_target)
        self.__update_target_network_polyak(self.hyperparameters.polyak,
                self.critic, self.critic_target)

    def __update_target_network_polyak(self, rho, inference_network, 
            target_network):
        """
        Updates targets networks weights as:
        new_weight = rho * old_weight + (1 - rho)*inference_network_weight.
        Where rho is the polyak hyperparameter.
        
        From https://discuss.pytorch.org/t/copying-part-of-the-weights/14199/5
        """
        params_inference = inference_network.named_parameters()
        params_target = target_network.named_parameters()
        params_target_dict = dict(params_target)
        # Let the dictionary hold the polyak-averaged value of the weights
        for weight_name, inference_weight_value in params_inference:
            if weight_name in params_target_dict:
                params_target_dict[weight_name].data.copy_(
                    rho*params_target_dict[weight_name].data
                    + (1 - rho)*inference_weight_value.data)
        
        target_network.load_state_dict(params_target_dict)

    def load_checkpoint(self, checkpoint_dir = None):
        if (checkpoint_dir == None):
            checkpoint_dir = self.hyperparameters.save_dir
        
        self.actor.load_checkpoint(os.path.join(checkpoint_dir, "actor"))
        self.actor_target.load_checkpoint(
                os.path.join(checkpoint_dir, "actor_target"))
        self.critic.load_checkpoint(os.path.join(checkpoint_dir, "critic"))
        self.critic_target.load_checkpoint(
            os.path.join(checkpoint_dir, "critic_target"))
        print("Agent: loaded checkpoint of network parameters at: "
                + checkpoint_dir)

    def __set_checkpoint_dir(self, pathname):
        self.checkpoint_dir = pathname
        self.actor.set_savefile_name(os.path.join(pathname, "actor"))
        self.actor_target.set_savefile_name(os.path.join(pathname, 
                "actor_target"))
        self.critic.set_savefile_name(os.path.join(pathname,"critic"))
        self.critic_target.set_savefile_name(os.path.join(pathname,
                "critic_target"))

    def save_checkpoint(self):
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()
        print("Agent: saved checkpoint of network parameters")
