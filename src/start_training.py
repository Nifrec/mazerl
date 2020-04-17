"""
Function (and helper functions) used to launch the training of a TD3 agent.

Author: Lulof Pirée
"""
# Library imports
import os
import torch
import gym
# Local imports
from agent.auxiliary import HyperparameterTuple, get_timestamp
from maze.environment_interface import Environment
from agent.trainer import Trainer
import settings
from main import Environments

def start_training(env: Environments):
    checkpoint_dir = __create_checkpoint_dir_if_needed()
    env = __create_environment(env)
    hyperparams = __create_hyperparameters(checkpoint_dir, env)
    logger = None
    print("TODO: add logger!")
    trainer = Trainer()
    trainer.setup_train_run()
    trainer.train()
    
def __create_checkpoint_dir_if_needed() -> str:
    path_to_here = os.path.dirname(__file__)
    checkpoint_dir = os.path.join(path_to_here, 
            settings.CHECKPOINT_TOP_DIR_NAME)
    if not os.path.exists(checkpoint_dir):
        print("Creating checkpint folder:\n" + checkpoint_dir)
        os.mkdir(checkpoint_dir)
    return checkpoint_dir

def __create_environment(env_name: str) -> "Gym-like environment":
    if (env_name == "gym-lunarlander"):
        return gym.make("LunarLanderContinuous-v2")
    elif (env_name == "maze"):
        return Environment()
    else:
        raise ValueError(f"Invalid environment name given:'{env_name}''")

def __create_hyperparameters(checkpint_dir: str, env:"Gym-like environment") \
        -> HyperparameterTuple:
    output = HyperparameterTuple(
            env = env,
            mode = settings.MODE, 
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            discount_rate=0.99,
            learning_rate_critic=0.0003,
            learning_rate_actor=0.0003,
            exploration_noise_std=0.1,
            min_action=-1.0,
            max_action=1.0,
            num_episodes=100000,
            batch_size=32,
            max_episode_duration=2000,
            memory_capacity=50000,
            polyak=0.999,
            # How many episodes between plot updates
            plot_interval=50, 
            moving_average_period=100,
            # How many episodes between saving data to disk
            checkpoint_interval=100, 
            action_size=2, # Length of action vector, depends on environment
            # relative directory name to save networks and plot in
            save_dir=os.path.join(checkpint_dir, get_timestamp() + "_" \
                    + settings.MODE),
            # Amount of initial episodes in which only random actions are taken.
            random_action_episodes=100,
            # The following hyperparameters are only used by TD3 (not by DDPG).
            # Noise added to action of target_actor during critic update.
            td3_critic_train_noise_std=0.2,
            # Noise will be clipped/clamped between this and negated this:
            td3_critic_train_noise_bound=0.5,
            # Amount of episodes between updating 
            # target critic- and actor- networks.
            td3_target_and_actor_update_interval=2
    )
    return output
