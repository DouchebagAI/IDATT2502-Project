import torch
import torch.nn as nn

class GoNN(nn.Module):
    def __init__(self, size=5):
        super().__init__()
        self.size = size
        self.dl1 = nn.Linear(size**2, size**3)
        self.dl2 = nn.Linear(size**3, size**3)
        self.output_layer = nn.Linear(size**3, size**2)

    def forward(self, x):
        x = self.dl1(x)
        x = torch.relu(x)
        x = self.dl2(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x
    


