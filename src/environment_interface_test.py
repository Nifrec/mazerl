"""
File to test the Environment class of environment_interface.py.

Author: Lulof Pirée
"""
# Library imports
import unittest
import numpy as np
import time
# Local imports
from environment_interface import Environment

def random_action():
    return np.random.normal(loc=0, scale=1, size=2)

def demonstrate_random_agent():
    """
    Performs a normal Reinforcement-Learning run, but with a non-learning
    random agent.
    """
    env = Environment()
    state = env.reset()
    done = False
    while not done:
        action = random_action()
        state, reward, done = env.step(action)
        print(f"action: {action}")
        print(f"State:\n{state}")
        print(f"Reward:{reward}")
        print(f"done:{done}")
        env.render()
        time.sleep(0.5)

class EnvironmentTestCase(unittest.TestCase):
    pass



if (__name__ == "__main__"):
    demonstrate_random_agent()
    unittest.main()