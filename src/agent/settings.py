"""
Various settings for the program launcher and training hyperparameters.

Author: Lulof Pirée
"""
# Library imports
import torch

# Local imports
from agent.auxiliary import Mode, Environments, get_timestamp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Parent directory of all checkpoint directories:
CHECKPOINT_TOP_DIR_NAME = "checkpoints"
# "TD3" or "DDPG".
MODE=Mode.TD3

# In/output sizes of actor and critic neural networks
LUNAR_CRITIC_IN = 8 + 2
LUNAR_ACTOR_IN = 8
LUNAR_CRITIC_OUT = 1
LUNAR_ACTOR_OUT = 2
MAZE_ACTOR_OUT = 2

# Hyperparameters:
BATCH_SIZE = 16
REPLAY_MEMORY_CAP = 10000
CHECKPOINT_INTERVAL=100

# Asynchronous mode hyperparameters:
SYNC_GRAD_INTERVAL = 10



