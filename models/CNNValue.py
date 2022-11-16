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
        # Conv
        # Relu
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
            nn.Flatten(),
            nn.Linear(1 * size ** 4, 1)
        )
        self.logits.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def f(self, x):
        return self.logits(x)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.l1_loss(self.logits(x), y)

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y).float())