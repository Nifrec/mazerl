from MazeGenerator import *
from conversions import *

import numpy as np


class Grid:
    """
    Creates a gridworld, where an expert (PolicyIterator) generates optimal and random
    demonstrations from which the IRL algorithm will then learn
    """

    def __init__(self, grid_size, epsilon=0.0):
        """
        :param grid_size: side len of the grid
        :param epsilon: exploration probability
        """
        self.grid_size = grid_size
        self.n_states = self.grid_size ** 2

        # Generate a maze
        print("Generating a maze:")
        maze_generator = MazeGenerator(self.grid_size)
        self.rewards = maze_generator.generate_maze()
        self.start_state = coord_to_int(maze_generator.maze_start_point, grid_size)
        self.goal_state  = coord_to_int(maze_generator.maze_end_point, grid_size)

        print("START STATE = ", self.start_state)
        print("GOAL STATE = ", self.goal_state)

        self.actions = np.array(["up", "down", "left", "right"])

        self.p_exploit = 1.0 - epsilon
        self.p_explore = epsilon / (len(self.actions) - 1)

    def get_feature(self, state):
        """
        Gets feature vector of a state
        :param state: int
        :return: vector: list
        """
        f = [0] * self.n_states
        f[state] = 1
        return f

    def get_rewards(self):
        return self.rewards.flatten()

    def result_of_action(self, state, action):
        """
        Performes given action on the state and gets the next state
        :param state: int
        :param action: string
        :return: int - next state
        """
        state_coord = int_to_coord(state, self.grid_size)

        next_states = [(max(0, state_coord[0] - 1), state_coord[1]),
                       (min(self.grid_size - 1, state_coord[0] + 1), state_coord[1]),
                       (state_coord[0], max(0, state_coord[1] - 1)),
                       (state_coord[0], min(self.grid_size - 1, state_coord[1] + 1))]

        # Probs for exploring
        transition_probabilities = self.p_explore * np.ones((len(self.actions)))
        # Probs for exploiting
        transition_probabilities[np.where(self.actions == action)[0][0]] = self.p_exploit

        # Selecting next state (performing "action" according to above probs)
        next_state = next_states[np.random.choice(range(len(next_states)), p=transition_probabilities)]

        return coord_to_int(next_state, self.grid_size)

    def generate_demos(self, policy=None, n_demos=10):
        """
        Generates a given number of demos for either optimal or random routes
        NOTE:: uncomment print statements to see demos
        :param policy: supplied for optimal deemos
        :param n_demos: int
        :return: np.array of feature vectors of demos
        """
        # Whether we're generating optimal demos or random ones
        is_optimal = not (policy is None)

        if not is_optimal:
            print("Generating non optimal demos...", end=" ")
            # Assigning random policy
            policy = np.random.choice(self.actions, size=self.n_states)
        else:
            print("Generating optimal demos...", end=" ")

        demos = []

        # For the optimal
        start_state = self.start_state
        goal_state = self.goal_state

        max_demo_len = self.grid_size * 3

        while len(demos) < n_demos:
            demo = []
            current_state = start_state

            if not is_optimal:
                # Randomize new endpoints for the new demo if not optimal
                current_state = np.random.randint(self.n_states)
                goal_state = np.random.randint(self.n_states)

            while current_state != goal_state and \
                    len(demo) < max_demo_len:

                # Add current state to the demo
                demo.append(self.get_feature(current_state))
                # Perform action according to a policy, get next state
                current_state = self.result_of_action(current_state, policy[current_state])

                #print(current_state, end="->")

            if (current_state == self.goal_state and is_optimal) or \
                    not is_optimal:
                demo.append(self.get_feature(self.goal_state))
                #print(current_state)

                # If successful demo, save it
                demos.append(np.array(demo))

        print("✓")
        return np.array(demos)

    def get_transition_probabilities(self, state, action):
        """
        Calculates transition probabilities for a given state and action
        Assumption: P(going over the border) = P(staying at the same state)
        :param state: int
        :param action: string
        :return: transition probabilities
        """

        transition_probabilities = np.zeros((self.grid_size, self.grid_size))
        state_coord = int_to_coord(state, self.grid_size)

        transition_probabilities[max(0, state_coord[0] - 1)][state_coord[1]] += \
            self.p_exploit if action == "up" else self.p_explore

        transition_probabilities[state_coord[0]][max(0, state_coord[1] - 1)] += \
            self.p_exploit if action == "left" else self.p_explore

        transition_probabilities[min(self.grid_size - 1, state_coord[0] + 1)][state_coord[1]] += \
            self.p_exploit if action == "down" else self.p_explore

        transition_probabilities[state_coord[0]][min(self.grid_size - 1, state_coord[1] + 1)] += \
            self.p_exploit if action == "right" else self.p_explore


        return transition_probabilities.flatten()

    def perform_greedy_action(self, values):
        """
        Performs greedy action
        :param values: q trained values
        :return: policy list
        """
        values = values.reshape(self.grid_size, self.grid_size)
        policy = np.repeat("random", self.n_states)

        for state in range(self.n_states):
            state_coord = int_to_coord(state, self.grid_size)
            policy[state] = self.actions[np.argmax([values[max(0, state_coord[0] - 1), state_coord[1]],
                                                    values[min(self.grid_size - 1, state_coord[0] + 1), state_coord[1]],
                                                    values[state_coord[0], max(0, state_coord[1] - 1)],
                                                    values[state_coord[0], min(self.grid_size - 1, state_coord[1] + 1)]])]
        return policy
