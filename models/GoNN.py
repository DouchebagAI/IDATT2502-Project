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
        return torch.softmax(self.logits(x), dim=1)

    def f_2(self, x):
        return self.logits(x)

    # Cross Entropy loss
    # A single greatest move
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x),  torch.abs(y).argmax(1))

    # MSE loss
    # A spread og how good each move is
    def mse_loss(self, x, y):
        return nn.functional.mse_loss(self.logits(x), torch.abs(y))


    def mse_acc(self, x, y):
        #print(torch.tensor([True if i in torch.topk(torch.abs(y), 3).indices else False for i in torch.topk(self.f(x), 3).indices]))
        return torch.mean(torch.tensor([True if i in torch.topk(torch.abs(y), 3).indices else False for i in torch.topk(self.f(x), 3).indices]).float())

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), torch.abs(y).argmax(1)).float())