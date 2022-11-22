import math
import random

from MCTSDNN.Node import Node, Type
from enum import Enum
import torch
import torch.nn as nn
import copy
import numpy as np
import gym

class GoDCNN(nn.Module):
    def __init__(self, size=3):
        super().__init__()
        self.size = size
        # Conv
        # Relu
        self.logits = nn.Sequential(
            nn.Conv2d(6, size**2, kernel_size=3, padding=2),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(size**2, size**3, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(size ** 3, size ** 4, kernel_size=3, padding=2),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(625, size**4),
            nn.Flatten(),
            nn.Linear(1*size**4, size**2+1)
        )
        self.logits.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    # Categorical (Only a single best move)
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x),  y.argmax(1))

    # MSE loss
    # A spread og how good each move is
    def mse_loss(self, x, y):
        return nn.functional.mse_loss(self.logits(x), y)

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())