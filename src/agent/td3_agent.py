"""
TD3 version of learning agent, subclass of Agent
Lulof Pirée (Nifrec)
"""
import torch
import numpy as np
from agent.agent_class import Agent
from agent.actor_network import ActorNetwork
from agent.critic_network import CriticNetwork

class TD3Agent(Agent):
    """
    Similar to the standard DDPG agent, but:
    * Uses a twin critic, and returns the minimum of their predictions.
    * Noise is added during updating the critics to the action of the
            target_actor. (Note that in DDPG, noise is only added to
            the action of the inference actor for actions actually performed)
    * The inference actor is updated when the targets are updated,
            not every time the critics are updated.

    See Fujimoto et al. 2018 for the TD3 paper
    ('Addressing Function Approximation Error in Actor-Critic Methods')
    """

    def __init__(self, hyperparameters, 
            actor_net: ActorNetwork, critic_net: CriticNetwork):
        super().__init__(hyperparameters, actor_net, critic_net)

    def compute_target_action(self, next_states):
        target_actions = self.actor_target.forward(next_states)
        target_actions += torch.clamp(
                torch.empty_like(target_actions).normal_(
                        mean=0,
                        std=self.hyperparameters.td3_critic_train_noise_std
                ),
                -1*self.hyperparameters.td3_critic_train_noise_bound,
                self.hyperparameters.td3_critic_train_noise_bound
        )
        return target_actions

    def update(self, batch, episode):
        """
        Takes a batch of Experience instances and uses them to update the
        inference networks using gradient ascent/descent.
        Arguments:
        * batch: iterable of Experience instances.
        * episode: int, current episode of training.

        Returns:
        * critic_loss, actor_loss: floats, losses (i.e. values used
                to optimize the networks) for critic and actor respectively.
                actor_loss may be None if it is not time to update the actor.
        """
        critic_loss = self.update_critic_net(batch).item()
        actor_loss = None
        if (episode % self.hyperparameters.td3_target_and_actor_update_interval\
                == 0):
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
        values_q1, values_q2 = \
                self.critic.forward(torch.cat((states, actions), dim=-1))
        # Compute Q-Value of next state + action (from target actor) according
        # to target critic. These are the 'good' examples.
        with torch.no_grad():
            target_actions = self.compute_target_action(next_states)
            target_val1, target_val2 = self.critic_target.forward(
                    torch.cat((next_states, target_actions), dim=-1))
            target_values = torch.min(target_val1, target_val2)
            targets = rewards.unsqueeze(dim=-1) \
                    + (dones == False) \
                    * self.hyperparameters.discount_rate * target_values

        # Perform backprop
        loss = ((targets - values_q1)**2).mean() \
                + ((targets - values_q2)**2).mean()
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        self.critic.eval()
        return loss
                
    def update_actor_net(self, batch):
        """
        Perform forward prop on actor, with as loss the predicted value
        of the predicted action in that state (not from target but from normal
        networks).
        """
        states, _, _, _, _ = self.extract_experiences(batch)
        self.critic.eval()
        actions = self.actor.forward(states)
        q1, q2 = self.critic.forward(torch.cat((states, actions), dim=-1))
        values = torch.min(q1, q2)
        self.actor.train()
        self.actor_optim.zero_grad()
        loss = ((-1*values).mean())
        loss.backward()
        self.actor_optim.step()
        self.actor.eval()

        return loss

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
        q1, q2 = self.critic.forward(torch.cat((state, action), dim=-1))
        return torch.min(q1, q2).item()