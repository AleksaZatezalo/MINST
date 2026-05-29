import torch.nn as nn

class MINSTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),   # 28*28=784 inputs
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # 10 Outputs: digits 0-9
        )

    def forward(self, x):
        return self.net(x)