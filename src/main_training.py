"""
Function (and helper functions) used to launch the training of a TD3 agent.

Author: Lulof Pirée
"""
# Library imports
import os
import torch
import gym
import enum
# Local imports
from agent.auxiliary import HyperparameterTuple, get_timestamp, setup_save_dir,\
        make_setup_info_file
from maze.environment_interface import Environment as MazeEnv
from agent.trainer import Trainer
from agent.logger import Logger
from agent.agent_class import Agent
import settings

class Environments(enum.Enum):
    maze = 1
    lunarlander = 2

def start_training(env: Environments):
    checkpoint_dir = __create_checkpoint_dir_if_needed()
    hyperparameters = __create_hyperparameters(checkpoint_dir, env)    
    __create_run_directory(hyperparameters)
    agent = Agent
    env = __create_environment(env)
    logger = Logger(hyperparameters)
    trainer = Trainer(hyperparameters, logger, agent)
    
    trainer.train()
    
def __create_checkpoint_dir_if_needed() -> str:
    """
    Creates the directory that holds the subdirectories of each
    recorded train run (if it does not exist already).
    Always returns the directory in which checkpoints directories
    should be stored.
    """
    path_to_here = os.path.dirname(__file__)
    checkpoint_dir = os.path.join(path_to_here, 
            settings.CHECKPOINT_TOP_DIR_NAME)
    if not os.path.exists(checkpoint_dir):
        print("Creating checkpint folder:\n" + checkpoint_dir)
        os.mkdir(checkpoint_dir)
    return checkpoint_dir

def __create_run_directory(hyperparameters: HyperparameterTuple):
    """
    Creates a directory in the checkpoints directory for this run.
    Here generated data such as parameters and performance stats will be logged.
    """
    print(f"Initialising {hyperparameters.mode} with"
            + f"self.hyperparameters: {hyperparameters}")
    setup_save_dir(hyperparameters.save_dir)
    make_setup_info_file(hyperparameters)\

def __create_environment(env_name: Environments) -> "Gym-like environment":
    if (env_name == Environments.lunarlander):
        return gym.make("LunarLanderContinuous-v2")
    elif (env_name == Environments.maze):
        return MazeEnv()
    else:
        raise ValueError(f"Invalid environment name given:'{env_name}''")

def __create_agent(self) -> Agent:
    assert False, "WIP"
    if (self.mode == "TD3"):
        self.agent = TD3Agent(self.hyperparameters, self.device,
                self.state_size, self.hyperparameters.save_dir)
    elif (self.mode == "DDPG"):
        self.agent = Agent(self.hyperparameters, self.device, 
                self.state_size, self.hyperparameters.save_dir)
    else:
        raise ValueError(self.__class__.__name__ + "invalid mode")

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
