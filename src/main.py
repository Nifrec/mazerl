#!/usr/bin/python3
"""
Main file giving a single access point to the project:
from this file the human interface can be launched 
(also possible through maze/human_interface.py),
and the TD3 algorithm can be used to train an agent,
or load a trained agent.

Author: Lulof Pirée
"""
import argparse
import os
import start_training
import enum

COMMANDS = ("human", "train", "load")
class Environments(enum.Enum):
    maze = 1
    lunarlander = 2
    
ENVIRONMENTS = ("gym-lunarlander", "maze")
HELP_COMMANDS = \
    """
    Function of the program to launch:
    * human: launches interactive maze with live keyboard input.
    * train: create a new agent and start training it.
    * load: load a checkpoint directory of a previously trained agent.
    """
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=COMMANDS, 
        help=HELP_COMMANDS)
parser.add_argument("-d", "--checkpoint-directory", required=False, 
        help="path to folder of saved agent (in case relevant)",
        metavar="path", dest="checkpoint_dir")
parser.add_argument("-e", "--environment", required=False, 
        help="environment to use for training (default: maze)",
        choices=ENVIRONMENTS, default="maze")
args = parser.parse_args()

if (args.command == "human"):
    from maze.human_interface import HumanInterface
    HumanInterface()
elif (args.command == "train"):
    print("Sorry, this feature is still WIP")
    if args.environment == "maze":
        start_training.start_training(Environments.maze)
    elif args.environment == "gym-lunarlander":
        start_training.start_training(Environments.lunarlander)

elif (args.command == "load"):
    print(args.checkpoint_dir)
    print("Sorry, this feature is still WIP")



