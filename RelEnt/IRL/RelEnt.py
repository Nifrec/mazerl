import numpy as np
import sys


class RelEnt:
    """
    Model-free Relative Entropy Inverse Reinforcement Learning
    """

    def __init__(self, optimal_demos, nonoptimal_demos):
        """
        :param optimal_demos: array of feature matrices of optimal demos
        :param nonoptimal_demos: array of feature matrices of not optimal demos
        """

        self.optimal_demos = optimal_demos
        self.nonoptimal_demos = nonoptimal_demos

        self.n_features = len(self.optimal_demos[0][0])
        self.weights = np.zeros((self.n_features,))
        self.optimals_feature = np.zeros_like(self.weights)
        self.policy_features = np.zeros((len(self.nonoptimal_demos), self.n_features))

    def calculate_objective(self):
        """
        For the partition function Z($\theta$), we just sum over all the exponents of their rewards, similar to
        the equation above equation (6) in the original paper.
        """
        objective = np.dot(self.optimals_feature, self.weights)
        for i in range(self.nonoptimal_demos.shape[0]):
            objective -= np.exp(np.dot(self.policy_features[i], self.weights))
        return objective

    def calculate_optimals_feature(self):
        """
        Calculates the expected feature for optimal's policy
        :return: expected feature
        """
        self.optimals_feature = np.zeros_like(self.weights)

        for i in range(len(self.optimal_demos)):
            self.optimals_feature += feature_averages(self.optimal_demos[i])
        self.optimals_feature /= len(self.optimal_demos)

        return self.optimals_feature

    def train(self, epochs=50000, step_size=1e-4, print_every=5000):
        """
        Train RelEnt
        """
        self.calculate_optimals_feature()
        self.policy_features = np.zeros((len(self.nonoptimal_demos), self.n_features))

        for i in range(len(self.nonoptimal_demos)):
            self.policy_features[i] = feature_averages(self.nonoptimal_demos[i])

        importance_sampling = np.zeros((len(self.nonoptimal_demos),))

        for epoch in range(epochs):
            sys.stdout.write('\r')
            percentage = (epoch+1) / epochs
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*percentage), 100*percentage))
            sys.stdout.flush()

            # Importance Sampling (4.3 from paper?)
            for i in range(len(self.nonoptimal_demos)):
                importance_sampling[i] = np.exp(np.dot(self.policy_features[i], self.weights))
            importance_sampling /= np.sum(importance_sampling, axis=0)

            # Weights for feature vectors
            weighted_sum = np.sum(np.multiply(np.array([importance_sampling, ] * self.policy_features.shape[1]).T,
                                              self.policy_features), axis=0)
            self.weights += step_size * (self.optimals_feature - weighted_sum)

            # One weird trick to ensure that the weights don't blow up the objective
            self.weights = self.weights / np.linalg.norm(self.weights, keepdims=True)


            if epoch % print_every == 0:
                #print("Value of objective is: " + str(self.calculate_objective()))
                pass



def feature_averages(demo, gamma=0.99):
    horizon = len(demo)
    return np.sum(np.multiply(demo,
                              np.array([gamma ** j for j in range(horizon)]).reshape(horizon, 1)), axis=0)
