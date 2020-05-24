import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.human_input.HAT_models.model import Model


class Convnet(Model):
    """
    Implementation based on the Model interface.
    Inspired by the Pytorch convnet implementation of adventuresinmachinelearning.com
    """

    # Convolutional neural network (two convolutional layers)
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.drop_out = nn.Dropout()
            self.fc1 = nn.Linear(7 * 7 * 64, 1000)
            self.fc2 = nn.Linear(1000, 10)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.drop_out(out)
            out = self.fc1(out)
            out = self.fc2(out)
            return out

    def __init__(self, training_data):
        # Hyperparameters
        self.num_epochs = 6
        self.num_classes = 10
        self.batch_size = 100
        self.learning_rate = 0.001

        # convert the training dataframe to a nested list of tuples by batch
        self.training_data = self.convert_to_proper_format(list(training_data.itertuples(index=False, name=None)))

        self.model_save_path = './convnet_models'

        self.model = self.Network()

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def convert_to_proper_format(self, data):
        """convert training data to nested list of tuples by batch"""
        new_data = []

        for i in range(math.floor(len(data)/self.batch_size)):
            temp = []
            for j in range(self.batch_size):
                temp.append(data[i * self.batch_size + j])
            new_data.append(temp)

        return new_data

    def train(self):
        """train the model"""
        steps = math.ceil(len(self.training_data))
        losses = []
        accuracies = []
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.training_data):
                # forward propagation
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                losses.append(loss.item())

                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # record the accuracy
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                accuracies.append(correct / total)

                # per 100 steps, print stats
                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, self.num_epochs, i + 1, steps, loss.item(),
                                  (correct / total) * 100))

        # save the model after training
        torch.save(self.model.state_dict(), self.model_save_path + 'conv_net_model.ckpt')

    def forward(self, state):
        """perform a forward pass of the model"""
        return self.model.forward(state)
