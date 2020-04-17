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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .auxiliary import Experience, HyperparameterTuple
from .network import Network
from .actor_network import ActorNetwork
from .logger import Logger


class Agent():
    """
    Class that keeps track of/does the following:
    * Store the neural networks.
    * Train the neural networks.
    * Select and clip actions.
    """
    CRITIC_CLASS = Network

    def __init__(self, hyperparameters, device, state_size, checkpoint_dir):
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
        self.device = device
        self.state_size = state_size

        self.create_networks()
        self.__set_checkpoint_dir(checkpoint_dir)

    def create_networks(self):        
        self.setup_critic_network()
        self.setup_actor_network()
        # Clone random initialization to target networks.
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_optim = optim.Adam(params=self.critic.parameters(),
                lr=self.hyperparameters.learning_rate_critic)
        self.actor_optim = optim.Adam(params=self.actor.parameters(), 
                lr=self.hyperparameters.learning_rate_actor)

    def setup_critic_network(self):
        actor_output_size = self.hyperparameters.action_size
        critic_output_size = 1 # The Q-value is just one number
        # in DDPG/TD3 Q-function is also function of the action -> Q(s, a)
        critic_input_size = self.state_size + actor_output_size
        self.critic = self.CRITIC_CLASS(critic_input_size, critic_output_size)\
                .to(self.device)
        self.critic_target = self.CRITIC_CLASS(critic_input_size, 
                critic_output_size).to(self.device)
        self.critic_target.eval() # Never gets trained, weights directly copied

    def setup_actor_network(self):
        actor_output_size = self.hyperparameters.action_size
        self.actor = ActorNetwork(self.state_size, actor_output_size)\
                .to(self.device)
        self.actor_target = ActorNetwork(self.state_size, actor_output_size)\
                .to(self.device)
        self.actor_target.eval() # Never gets trained, weights directly copied

    def choose_action(self, state, select_random=False):
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
                stored on self.device.
        * estimation: float, estimated Q-value for this state-action pair.
        """
        assert (state.shape[-1] == self.state_size)
        if (select_random):
            action = (torch.rand(self.hyperparameters.action_size)-0.5)*2
            action = action.unsqueeze(0).to(self.device)
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
        return self.critic.forward(torch.cat((state, action), dim=-1)).item()

    def update(self, batch, episode):
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
        critic_loss = self.update_critic_net(batch).item()
        actor_loss = self.update_actor_net(batch).item()
        self.update_target_networks()
        return critic_loss, actor_loss
                
    def update_critic_net(self, batch):
        states, actions, rewards, next_states, dones \
            = self.extract_experiences(batch)
        self.critic.train()
        self.actor.eval()
        
        # Current Q-Value prediction based on historical state-action pair
        # = 'bad' thing to improve
        # These can be different than at time of gathering the experience.
        # This does not matter and is a feature not a bug!
        values = self.critic.forward(torch.cat((states, actions), dim=-1))
        # Compute Q-Value of next state + action (from target actor) according
        # to target critic. These are the 'good' examples.
        target_actions = self.compute_target_action(next_states)
        target_values = self.critic_target.forward(
                torch.cat((next_states, target_actions), dim=-1))
        targets = rewards.unsqueeze(dim=-1) \
                + (dones == False) \
                    * self.hyperparameters.discount_rate * target_values
        
        # Perform backprop
        self.critic_optim.zero_grad()
        loss = ((targets - values)**2).mean()
        loss.backward()
        self.critic_optim.step()
        self.critic.eval()
        return loss

    def compute_target_action(self, next_states):
        return self.actor_target.forward(next_states)

    def update_actor_net(self, batch):
        """
        Perform forward prop on actor, with as loss the predicted value
        of the predicted action in that state (not from target but from normal
        networks).
        """
        states, _, _, _, _ = self.extract_experiences(batch)
        self.critic.eval()
        actions = self.actor.forward(states)
        values = self.critic.forward(torch.cat((states, actions), dim=-1))
        self.actor.train()
        self.actor_optim.zero_grad()
        loss = ((-1*values).mean())
        loss.backward()
        self.actor_optim.step()
        self.actor.eval()

        return loss

    def extract_experiences(self, experiences):
        """
        Takes a batch of experiences and splits it into 4 tensors,
        each containing either the states, awards, rewards or 
        next_states of all experiences.

        Arguments:
        * experiences: list of Experience objects.

        Returns:
        A tuple with:
        * states: rank-2 tensor of states of all experiences,
            shape (num_experiences, 4).
        * actions: rank-1 tensor of actions of all experiences.
        * rewards: rank-1 tensor of rewards of all experiences.
        * next_states: rank-1 tensor of next_states of all experiences.
        """
        batch = Experience(*zip(*experiences))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)

        # Detach copies without the computational graph.
        # Without this call the old emptied computational graphs cause trouble.
        states = states.detach()
        actions = actions.detach()
        rewards = rewards.detach()
        next_states = next_states.detach()
        dones = dones.detach()

        return (states, actions, rewards, next_states, dones)

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