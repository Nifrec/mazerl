"""
TD3 critic-network implementation, subclass of Network.
Where DDPG uses a single neural network as a critic,
TD3 uses two identical networks. They are represented by a single
instance of the TwinNetwork class.

Author: Lulof Pirée (Nifrec) 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .network import Network

HIDDEN_LAYER_1_SIZE = 256
HIDDEN_LAYER_2_SIZE = 256
HIDDEN_LAYER_3_SIZE = 256

class TwinNetwork(Network):
    """
    Class to implement a twin critic (Q-value) neural network.
    Uses two identical critics, but always returns the minimum
    of the predictions of the two twins.
    """

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def create_layers(self, input_size, output_size):
        self.q1_l1 = nn.Linear(in_features=input_size,
                out_features=HIDDEN_LAYER_1_SIZE)
        self.q1_l2 = nn.Linear(in_features=HIDDEN_LAYER_1_SIZE,
                out_features=HIDDEN_LAYER_2_SIZE)
        self.q1_l3 = nn.Linear(in_features=HIDDEN_LAYER_2_SIZE,
                out_features=HIDDEN_LAYER_3_SIZE)
        self.q1_out = nn.Linear(in_features=HIDDEN_LAYER_3_SIZE,
                out_features=output_size)

        self.q2_l1 = nn.Linear(in_features=input_size,
                out_features=HIDDEN_LAYER_1_SIZE)
        self.q2_l2 = nn.Linear(in_features=HIDDEN_LAYER_1_SIZE,
                out_features=HIDDEN_LAYER_2_SIZE)
        self.q2_l3 = nn.Linear(in_features=HIDDEN_LAYER_2_SIZE,
                out_features=HIDDEN_LAYER_3_SIZE)
        self.q2_out = nn.Linear(in_features=HIDDEN_LAYER_3_SIZE,
                out_features=output_size)

    def forward(self, t):
        # Let each of the twins 'q1' and 'q2' make
        # a prediction, and then returen the minimum of the two.
        t_q1 = t.clone()
        t_q1 = F.leaky_relu(self.q1_l1(t_q1))
        t_q1 = F.leaky_relu(self.q1_l2(t_q1))
        t_q1 = F.leaky_relu(self.q1_l3(t_q1))
        t_q1 = F.leaky_relu(self.q1_out(t_q1))

        t_q2 = t.clone()
        t_q2 = F.leaky_relu(self.q2_l1(t_q2))
        t_q2 = F.leaky_relu(self.q2_l2(t_q2))
        t_q2 = F.leaky_relu(self.q2_l3(t_q2))
        t_q2 = F.leaky_relu(self.q2_out(t_q2))

        return t_q1, t_q2