"""
Function (and helper functions) used to launch the training of a TD3 agent.

Author: Lulof Pirée
"""
# Library imports
import os
import torch
import gym
import enum
from typing import Any, Tuple
# Local imports
from agent.auxiliary import HyperparameterTuple, get_timestamp, setup_save_dir,\
        make_setup_info_file, Mode
from agent.greyscale_env_wrapper import GreyScaleEnvironment as MazeEnv
from agent.trainer import Trainer
from agent.logger import Logger
from agent.agent_class import Agent
from agent.td3_agent import TD3Agent
from agent.actor_network import ActorNetwork
from agent.critic_network import CriticNetwork
from agent.actor_network_convolutional import ActorCNN
from agent.critic_network_convolutional import CriticCNN
import agent.settings as settings

class Environments(enum.Enum):
    maze = 1
    lunarlander = 2

def start_training(env_name: Environments):
    checkpoint_dir = __create_checkpoint_dir_if_needed()
    env = __create_environment(env_name)
    hyperparameters = __create_hyperparameters(checkpoint_dir, env)    
    __create_run_directory(hyperparameters)
    actor, critic = __create_networks(env_name)
    agent = __create_agent(hyperparameters, checkpoint_dir, actor, critic)
    
    logger = Logger(hyperparameters)
    trainer = Trainer(hyperparameters, agent, logger)
    
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

def __create_environment(env_name: Environments) -> Any:
    if (env_name == Environments.lunarlander):
        return gym.make("LunarLanderContinuous-v2")
    elif (env_name == Environments.maze):
        return MazeEnv()
    else:
        raise ValueError(f"Invalid environment name given:'{env_name}''")

def __create_agent(hyperparameters: HyperparameterTuple, 
        checkpoint_dir: str, actor: ActorNetwork,critic:CriticNetwork) -> Agent:
    agent: Agent

    if (settings.MODE == Mode.TD3):
        agent = TD3Agent(hyperparameters, actor, critic)
    elif (settings.MODE == Mode.DDPG):
        agent = Agent(hyperparameters, actor, critic)
    else:
        raise ValueError(f"invalid mode '{settings.MODE}")

    return agent

def __create_networks(env_name: Environments) \
        -> Tuple[ActorNetwork, CriticNetwork]:
    if (env_name == Environments.lunarlander):
        actor_net = ActorNetwork(settings.LUNAR_ACTOR_IN,
                settings.LUNAR_ACTOR_OUT)
        critic_net = CriticNetwork(settings.LUNAR_CRITIC_IN,
                settings.LUNAR_CRITIC_OUT, mode=settings.MODE)
    elif (env_name == Environments.maze):
        # Input and output sizes ignored by the CNNs
        actor_net = ActorCNN(0, 0)
        critic_net = CriticCNN(0, 0, mode=settings.MODE)
    else:
        raise ValueError(f"invalid environment '{env_name}")

    actor_net.to(settings.DEVICE)
    critic_net.to(settings.DEVICE)

    return actor_net, critic_net

def setup_critic_network(self):
    assert False,  "WIP"
    actor_output_size = self.hyperparameters.action_size
    critic_output_size = 1 # The Q-value is just one number
    # in DDPG/TD3 Q-function is also function of the action -> Q(s, a)
    critic_input_size = self.state_size + actor_output_size
    self.critic = self.CRITIC_CLASS(critic_input_size, critic_output_size)\
            .to(self.device)
    self.critic_target = self.CRITIC_CLASS(critic_input_size, 
            critic_output_size).to(self.device)
        

def setup_actor_network(self):
    assert False, "WIP"
    actor_output_size = self.hyperparameters.action_size
    self.actor = ActorNetwork(self.state_size, actor_output_size)\
            .to(self.device)
    self.actor_target = ActorNetwork(self.state_size, actor_output_size)\
            .to(self.device)

def __create_hyperparameters(checkpint_dir: str, env:Any) \
        -> HyperparameterTuple:
    output = HyperparameterTuple(
            env = env,
            mode = settings.MODE, 
            device=settings.DEVICE,
            discount_rate=0.99,
            learning_rate_critic=0.0003,
            learning_rate_actor=0.0003,
            exploration_noise_std=0.1,
            min_action=-1.0,
            max_action=1.0,
            num_episodes=100000,
            batch_size=settings.BATCH_SIZE,
            max_episode_duration=2000,
            memory_capacity=settings.REPLAY_MEMORY_CAP,
            polyak=0.999,
            # How many episodes between plot updates
            plot_interval=50, 
            moving_average_period=100,
            # How many episodes between saving data to disk
            checkpoint_interval=settings.CHECKPOINT_INTERVAL, 
            action_size=2, # Length of action vector, depends on environment
            # relative directory name to save networks and plot in
            save_dir=os.path.join(checkpint_dir, get_timestamp() + "_" \
                    + str(settings.MODE)),
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
