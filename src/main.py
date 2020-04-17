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
import settings

def start_training():
    path_to_main = os.path.dirname(__file__)
    checkpoint_dir = os.path.join(path_to_main, settings.CHECKPOINT_TOP_DIR_NAME)
    if not os.path.exists(checkpoint_dir):
        print("Creating checkpint folder:\n" + checkpoint_dir)
        os.mkdir(checkpoint_dir)

COMMANDS = ("human", "train", "load")
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=COMMANDS, 
        help="Function of the program to launch")
parser.add_argument("-d", "--checkpoint-directory", required=False, 
        help="path to folder of saved agent (in case relevant)",
        metavar="path", dest="checkpoint_dir")
args = parser.parse_args()

if (args.command == "human"):
    from maze.human_interface import HumanInterface
    HumanInterface()
elif (args.command == "train"):
    print("Sorry, this feature is still WIP")
    start_training()
elif (args.command == "load"):
    print(args.checkpoint_dir)
    print("Sorry, this feature is still WIP")



