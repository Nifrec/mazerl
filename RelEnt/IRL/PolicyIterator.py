import numpy as np


class PolicyIterator:
    """
    Policy iteration for simulating optimal demos
    """
    def __init__(self, env):
        self.env = env
        self.values = np.zeros((self.env.n_states,))
        self.policy = np.random.choice(self.env.actions, size=self.env.n_states)

    def policy_evaluation(self, epochs=10, gamma=0.99):
        for _ in range(epochs):

            transition_probabilities = np.zeros((self.env.n_states, self.env.n_states))
            for state in range(self.env.n_states):
                transition_probabilities[state] = self.env.get_transition_probabilities(state, self.policy[state])

            self.values = self.env.get_rewards() + \
                          gamma * np.dot(transition_probabilities, self.values)

    def policy_iteration(self, epochs=10):
        for _ in range(epochs):
            self.policy_evaluation()
            self.policy = self.env.perform_greedy_action(self.values)
        return self.policy
