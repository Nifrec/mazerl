"""
Same as Network in network.py,
but with an additional tanh in output layer.

Lulof Pirée
"""

import torch
from .network import Network

class ActorNetwork(Network):

    def forward(self, t):
        # Need to clone() otherwise pytorch complaints about
        # having in-place operations in autograd's tensors.
        return torch.tanh(super().forward(t)).clone();