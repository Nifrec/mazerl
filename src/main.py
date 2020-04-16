"""
Main file giving a single access point to the project:
from this file the human interface can be launched 
(also possible through maze/human_interface.py),
and the TD3 algorithm can be used to train an agent,
or load a trained agent.

Author: Lulof Pirée
"""
import sys
import os
sys.path.append(os.path.abspath("./maze"))
sys.path.append(os.path.abspath("./agent"))
from maze import *
from agent import *
from maze import model, human_interface, environment_interface, distances

from maze.human_interface import HumanInterface

HumanInterface()