from MazeGenerator import *
from conversions import *

import numpy as np
import sys
import matplotlib.pyplot as plt


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
        self.maze_generator = MazeGenerator(self.grid_size)
        self.rewards = self.maze_generator.generate_maze()
        self.start_state = coord_to_int(self.maze_generator.maze_start_point, grid_size)
        self.goal_state  = coord_to_int(self.maze_generator.maze_end_point, grid_size)

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

    def get_optimal_demo(self, optimal_policy):
        demo = []
        current_state = self.start_state

        while current_state != self.goal_state:
            # Add current state to the demo
            demo.append (current_state)
            # Perform action according to a policy, get next state
            current_state = self.result_of_action (current_state, optimal_policy[current_state])

        if current_state == self.goal_state:
            demo.append (current_state)

        return demo

    def generate_maze_image(self):
        fig, ax = plt.subplots (nrows=1, ncols=1, figsize=(self.grid_size, self.grid_size))

        c1 = ax.pcolor (self.rewards, edgecolors='k', linewidths=2)
        plt.colorbar (c1, ax=ax)

        offset = 0.2

        for state in range(self.grid_size**2):
            state_cell = int_to_coord(state, self.grid_size)
            x = state_cell[1] + offset
            y = state_cell[0] + offset
            plt.text(x, y, str(state),
                     fontsize=20, fontweight=30, color="red")

        fig.show ()

    def request_human_demos(self, optimal_policy=None, n_demos=10):

        if optimal_policy is not None:
            # Provide a human with the optimal Q trajectory
            optimal_demo = self.get_optimal_demo(optimal_policy)
            print("\nSuggested optimal demo:\n", optimal_demo)

        demos = []

        for i in range(n_demos):
            # Ask for a new demo
            progress = str(i+1) + "/" + str(n_demos) + ": "
            demo_string = input(progress)
            demo = [self.get_feature(int(state)) for state in demo_string.split()]
            demos.append(demo)

        return demos

    def get_random_start(self, random_percentage):
        """
        Selects a random cells in a maze to start a demo from
        NOTE:: it actually doesnt learn well when not starting from the same point all the time
        :param random_percentage: [0,1] how often you wanna select random start
        :return: selected state
        """
        if np.random.uniform() <= (1 - random_percentage):
            return self.start_state
        while True:
            start_state = np.random.randint(0, self.grid_size**2 - 1)
            start_cell = int_to_coord(start_state, self.grid_size)
            if start_cell in self.maze_generator.maze_cells or \
               start_cell in self.maze_generator.distraction_cells:
                return start_state

    def generate_demos(self, policy=None, n_demos=10, random_start=(True, 0.1)):
        """
        Generates a given number of demos for either optimal or random routes
        NOTE:: uncomment print statements to see demos
        :param policy: supplied for optimal demos
        :param n_demos: int
        :random_start: bool
        :return: np.array of feature vectors of demos
        """
        # Whether we're generating optimal demos or random ones
        is_optimal = not (policy is None)

        if not is_optimal:
            # Assigning random policy
            policy = np.random.choice(self.actions, size=self.n_states)

        demos = []
        demo_id = 0

        # For the optimal
        goal_state = self.goal_state

        max_demo_len = self.grid_size * 10

        while len(demos) < n_demos:
            sys.stdout.write ('\r')
            percentage = (demo_id + 1) / n_demos
            sys.stdout.write ("[%-20s] %d%%" % ('=' * int (20 * percentage), 100 * percentage))
            sys.stdout.flush ()

            demo = []
            if random_start[0]:
                current_state = self.get_random_start (random_start[1])
            else:
                current_state = self.start_state

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
                demo.append(self.get_feature(current_state))
                #print(current_state)

                # If successful demo, save it
                demos.append(np.array(demo))
                demo_id += 1

        print(" ✓")
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
