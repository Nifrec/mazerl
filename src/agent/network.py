"""
Simple neural network of fully-connected layers and ReLU activation.
Used for actor/critic networks, and as base class for networks
compatible with Agent and Trainer.

Author: Lulof Pirée
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

HIDDEN_LAYER_1_SIZE = 256
HIDDEN_LAYER_2_SIZE = 256
HIDDEN_LAYER_3_SIZE = 256

class Network(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.create_layers(input_size, output_size)
        
        # String used for checkpoint files.
        self.savefile_name = None

        # Public fields
        self.input_size = input_size
        self.output_size= output_size

    def create_layers(self, input_size, output_size):
        self.l1 = nn.Linear(in_features=input_size, 
                out_features=HIDDEN_LAYER_1_SIZE)
        self.l2 = nn.Linear(in_features=HIDDEN_LAYER_1_SIZE,
                out_features=HIDDEN_LAYER_2_SIZE)
        self.l3 = nn.Linear(in_features=HIDDEN_LAYER_2_SIZE, 
                out_features=HIDDEN_LAYER_3_SIZE)
        self.out = nn.Linear(in_features=HIDDEN_LAYER_3_SIZE,
                out_features=output_size)
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.relu(self.l1(t))
        t = torch.relu(self.l2(t))
        t = torch.relu(self.l3(t))
        t = self.out(t)
        return t

    def set_savefile_name(self, name):
        self.savefile_name = name

    def load_checkpoint(self, filename=None):
        if (filename == None):
            self.load_state_dict(torch.load(self.savefile_name))
        else:
            self.load_state_dict(torch.load(filename))

    def save_checkpoint(self, filename=None):
        if (filename == None):
            torch.save(self.state_dict(), self.savefile_name)
        else:
            torch.save(self.state_dict(), filename)