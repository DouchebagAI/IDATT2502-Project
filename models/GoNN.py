import math
import random

from MCTSDNN.Node import Node, Type
from enum import Enum
import torch
import torch.nn as nn
import copy
import numpy as np
import gym

#0.79
class GoNN(nn.Module):
    def __init__(self, size=3, kernel_size = 3):
        """
        Creates instance of model
        Uses single layers of 2D convolutional networks
        Flattens, then does two Linear transformations until desired output size (size**2 + 1)
        Uses RELU and MaxPool2d
        Uses CUDA if available
        """
        super().__init__()
        self.size = size
        lin = 100 if kernel_size == 5 else 225
        # Conv
        # Relu
        self.logits = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(6, size**2, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(225, size**4),
            nn.Linear(1*size**4, size**2+1)
        )
        self.logits.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def f(self, x):
        """
        :return: output with softmax
        """
        return torch.softmax(self.logits(x), dim=1)

    def f_2(self, x):
        """
        :return: output without softmax
        """
        return self.logits(x)

    def loss(self, x, y):
        """
        This is a function for cross entropy loss (CE).
        A single greatest move.

        :param x: model input
        :param y: model target
        :return: loss
        """
        return nn.functional.cross_entropy(self.logits(x),  torch.abs(y).argmax(1))

    def mse_loss(self, x, y):
        """
        This is a function for mean sqaure error (MSE)
        A spread of how good each move is.

        :param x: model input
        :param y: model target
        :return: loss
        """
        return nn.functional.mse_loss(self.logits(x), torch.abs(y))


    def mse_acc(self, x, y):
        """
        Checks if the 3 best moves in the model target and the actual target matches
        
        :param x: model input
        :param y: model target
        :return: accuracy
        """
        return torch.mean(torch.tensor([True if i in torch.topk(torch.abs(y), 3).indices else False for i in torch.topk(self.f(x), 3).indices]).float())

    # Accuracy
    def accuracy(self, x, y):
        """
        Checks If the best move is the same for model target, and actual target

        :param x: model input
        :param y: model target
        :return: accuracy
        """
        return torch.mean(torch.eq(self.f(x).argmax(1), torch.abs(y).argmax(1)).float())