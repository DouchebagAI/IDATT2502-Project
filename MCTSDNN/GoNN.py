import torch
import torch.nn as nn

class GoNN(nn.Module):
    def __init__(self, size=5):
        super().__init__()
        self.size = size
        self.dl1 = nn.Linear(size*size, size*size*size)
        