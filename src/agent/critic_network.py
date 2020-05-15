"""
TD3 critic-network implementation, subclass of Network.
Where DDPG uses a single neural network as a critic,
TD3 uses two identical networks. They are represented by a single
instance of the TwinNetwork class. Backwards compatible with DDPG by setting
mode = Mode.DDPG during initialization.

Author: Lulof Pirée (Nifrec) 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.agent.network import Network
from src.agent.auxiliary import Mode

HIDDEN_LAYER_1_SIZE = 256
HIDDEN_LAYER_2_SIZE = 256
HIDDEN_LAYER_3_SIZE = 256

class CriticNetwork(Network):
    """
    Class to implement a twin critic (Q-value) neural network.
    Uses two identical critics, the idea is to use the minimum
    of the predictions of the two twins. Note that both outputs should
    be used in optimization.
    """

    def __init__(self, input_size: int, output_size: int, mode: Mode):
        super().__init__(input_size, output_size)
        self.mode = mode

    def create_layers(self, input_size:int, output_size:int):
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

    def forward(self, state:torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Returns output of foward propagation.
        In DDPG mode returns only one value,
        in TD3 mode returns values of both critics.
        """
        t = torch.cat((state, action), dim=-1)
        # Let each of the twins 'q1' and 'q2' make
        # a prediction, and then returen the minimum of the two.
        t_q1 = t.clone()
        t_q1 = F.leaky_relu(self.q1_l1(t_q1))
        t_q1 = F.leaky_relu(self.q1_l2(t_q1))
        t_q1 = F.leaky_relu(self.q1_l3(t_q1))
        t_q1 = F.leaky_relu(self.q1_out(t_q1))

        if (self.mode == Mode.DDPG):
            return t_q1

        t_q2 = t.clone()
        t_q2 = F.leaky_relu(self.q2_l1(t_q2))
        t_q2 = F.leaky_relu(self.q2_l2(t_q2))
        t_q2 = F.leaky_relu(self.q2_l3(t_q2))
        t_q2 = F.leaky_relu(self.q2_out(t_q2))

        return t_q1, t_q2