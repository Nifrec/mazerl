"""
File to test the Environment class of environment_interface.py.
Has following functions:
* demonstrate_random_agent() shows how an random agent interacts with the
    environment and prints the oberservations, rewards, etc.
* show_rgb_observation() shows an observation RGB array as returned by the 
    Environment as an image. (used to confirm that the observations are 
    as expected.)


Author: Lulof Pirée
"""
# Library imports
import numpy as np
import time
import matplotlib
#matplotlib.use('agg')
import cv2
import matplotlib.pyplot as plt
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

def show_rgb_observation():
    """
    Shows the RGB array outputted by the Environment as observation,
    as an image on the screen.
    """
    env = Environment()
    observation = env.reset()
    width = len(observation[0])
    height = len(observation[0][0])
    channels = 3 # One for red, for green and for blue
    # Observation is a channel x width x height array,
    # Matplotlib expects a height x width x channel array
    # NOTE: np.reshape() does NOT work here! 
    # (probably uses the nearest numbers of teh same channel,
    # to form RGB triplets, instead of neighbours across channels).
    reshaped_observation = np.moveaxis(observation, 0, -1)
    reshaped_observation = np.moveaxis(reshaped_observation, 1, 0)
    plt.imshow(reshaped_observation.copy(), vmin=0, vmax=255)
    plt.show()

if (__name__ == "__main__"):
    #show_rgb_observation()
    demonstrate_random_agent()