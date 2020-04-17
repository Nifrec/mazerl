# mazerl
Maze environment for reinforcement learning (RL) with a continous action space
Provides an interface similar to OpenAI, and uses pygame to render the environment.

Provides the following modules:
`src/maze` provides the complete maze environment, including a human interface.
`src/agent` provides an implementation of DDPG and TD3 to train an agent for the maze using Deep Reinforcement Learning.
`src/main.py` is an command-line command that can be used to launch the training or the human interface.

## human-interface
`src/maze/human_interface.py` provides a human-playable script to play with the environment yourself. Simply move the ball with the arrowkeys (or WASD), F1 to generate a new maze, and ESC to exit. When you die you receive briefly a red screen, and when you reach the end you briefly receive a green screen, and in both cases a new maze is generated (note that the interface for AIs does not show the win/fail screens). The script can be lauched as `main.py human` or by running `human_interface.py`.
