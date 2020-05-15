
"""
Main file giving a single access point to the project:
from this file the human interface can be launched 
(also possible through maze/human_interface.py),
and the TD3 algorithm can be used to train an agent,
or load a trained agent.

Author: Lulof Pirée
"""
import argparse
# Local imports
import src.main_training

COMMANDS = ("human", "train", "load")
    
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
parser.add_argument('-a', "--asynchronous", action='store_true',
        help="Use Asynchronous Methods during training." \
            +" Only used for 'train' command.")
args = parser.parse_args()

if (args.command == "human"):
    from maze.human_interface import HumanInterface
    HumanInterface()
elif (args.command == "train"):
    asynchronous = args.asynchronous
    if args.environment == "maze":
        src.main_training.start_training(src.main_training.Environments.maze,
                asynchronous)
    elif args.environment == "gym-lunarlander":
        src.main_training.start_training(src.main_training.Environments.lunarlander,
                asynchronous)

elif (args.command == "load"):
    print(args.checkpoint_dir)
    print("Sorry, this feature is still WIP")



