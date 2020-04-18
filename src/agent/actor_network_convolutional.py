"""
Convolutional actor network build for the MazeRL environment.

Lulof Pirée
"""

import torch
import torch.nn as nn
from agent.actor_network import ActorNetwork
from maze.environment_interface import WINDOW_HEIGHT, WINDOW_WIDTH

INPUT_CHANNELS = 3
CONV_LAYERS_OUT = 12 * (WINDOW_HEIGHT * WINDOW_WIDTH) // (6*4) # = 768000
L1_OUT = 1000
L2_OUT = 300
L3_OUT = 1

class ActorCNN(ActorNetwork):
    
    def create_layers(self, input_size, output_size):
        self.cnn_layers = nn.Sequential(
            # 3 x 800 x 480
            nn.Conv2d(in_channels=INPUT_CHANNELS,
                    out_channels=2*INPUT_CHANNELS, kernel_size = 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=6, stride=6),
            # 6 x (800 x 480) / 3
            nn.Conv2d(in_channels=2*INPUT_CHANNELS,
                    out_channels=4*INPUT_CHANNELS, kernel_size = 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)
            # 12 x (800 x 480) / (3*2)
        )

        self.linear = nn.Sequential(
                nn.Linear(CONV_LAYERS_OUT, L1_OUT),
                nn.ReLU(inplace=True),
                nn.Linear(L1_OUT, L2_OUT),
                nn.ReLU(inplace=True),
                nn.Linear(L2_OUT, L3_OUT)
        )

    def forward(self, t):
        t = self.cnn_layers(t)
        t = self.linear(t.view(-1, CONV_LAYERS_OUT))
        # Need to clone() otherwise pytorch complaints about
        # having in-place operations in autograd's tensors.
        return torch.tanh(super().forward(t)).clone();