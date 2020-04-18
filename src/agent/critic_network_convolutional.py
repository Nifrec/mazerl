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

from agent.critic_network import CriticNetwork
from agent.auxiliary import Mode
from maze.environment_interface import WINDOW_HEIGHT, WINDOW_WIDTH

INPUT_CHANNELS = 3
CONV_LAYERS_OUT = 12 * (WINDOW_HEIGHT * WINDOW_WIDTH) / (3*2) # = 768000
L1_OUT = 1000
L2_OUT = 300
L3_OUT = 1

class CriticCNN(CriticNetwork):
    """
    Class to implement a twin critic (Q-value) neural network.
    Uses two identical critics, the idea is to use the minimum
    of the predictions of the two twins. Note that both outputs should
    be used in optimization.
    """

    def __init__(self, input_size: int, output_size: int, mode: Mode):
        super().__init__(input_size, output_size, mode)

    def create_layers(self, input_size:int, output_size:int):
        self.q1 = nn.Sequential(
            # 3 x 800 x 480
            nn.Conv2d(in_channels=INPUT_CHANNELS,
                    out_channels=2*INPUT_CHANNELS, kernel_size = 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            # 6 x (800 x 400) / 3
            nn.Conv2d(in_channels=2*INPUT_CHANNELS,
                    out_channels=4*INPUT_CHANNELS, kernel_size = 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 12 x (800 x 400) / (3*2)
            nn.Linear(CONV_LAYERS_OUT, L1_OUT),
            nn.LeakyReLU(inplace=True),
            nn.Linear(L1_OUT, L2_OUT),
            nn.LeakyReLU(inplace=True),
            nn.Linear(L2_OUT, L3_OUT)
        )

        self.q2 = nn.Sequential(
            # 3 x 800 x 480
            nn.Conv2d(in_channels=INPUT_CHANNELS,
                    out_channels=2*INPUT_CHANNELS, kernel_size = 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            # 6 x (800 x 400) / 3
            nn.Conv2d(in_channels=2*INPUT_CHANNELS,
                    out_channels=4*INPUT_CHANNELS, kernel_size = 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 12 x (800 x 400) / (3*2)
            nn.Linear(CONV_LAYERS_OUT, L1_OUT),
            nn.LeakyReLU(inplace=True),
            nn.Linear(L1_OUT, L2_OUT),
            nn.LeakyReLU(inplace=True),
            nn.Linear(L2_OUT, L3_OUT)
        )

    def forward(self, t: torch.Tensor):
        """
        Returns output of foward propagation.
        In DDPG mode returns only one value,
        in TD3 mode returns values of both critics.
        """
        # Let each of the twins 'q1' and 'q2' make
        # a prediction, and then returen the minimum of the two.
        t_q1 = t.clone()
        t_q1 = self.q1(t_q1)

        if (self.mode == Mode.DDPG):
            return t_q1

        t_q2 = t.clone()
        t_q2 = self.q2(t_q2)

        return t_q1, t_q2