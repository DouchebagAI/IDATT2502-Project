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
class GoCNNValue(nn.Module):
    def __init__(self, size=3, kernel_size = 3):
        super().__init__()
        self.size = size
        lin = 100 if kernel_size == 5 else 225
        """
        Creates model with two 2d convd layers
        Uses MaxPool2d and Relu
        Flattens, then does two Linear transformations until desired output size (size**2 + 1)
        Uses RELU and MaxPool2d
        Uses CUDA if available
        """
        self.logits = nn.Sequential(
            nn.Conv2d(6, size ** 2, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(size ** 2, size ** 3, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(500, size ** 4),
            nn.Linear(1 * size ** 4, 1)
        )

        self.logits.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def f(self, x):
        """
        :return: output with softmax
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
        return nn.functional.l1_loss(self.logits(x), y)


    def accuracy(self, x, y):
        """
        Measures accuracy by checking if model and target y model both round to same int (loss, draw, win)

        :param x: model input
        :param y: model target
        :return: accuracy
        """
        return torch.mean(torch.eq(torch.round(self.f(x).argmax(1).float())
                                        , torch.round(y).float()).float())
        