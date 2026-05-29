"""
Description: An implementation of the MINST classifcation problem in Pytorch.
Author: Aleksa Zatezalo
Date: May 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Dummy Data
x = torch.tensor(([4,6,2,-1,6,2,5],[1,6,2,-6,5,-3,5]))
print(x)

# ReLU example
R = nn.ReLU()
print(R(x))

# Plotting
x = torch.linspace(-3,3,100)
y = R(x)
plt.plot(x.numpy(), y.numpy())
plt.grid()
plt.show()

# Neural Network
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(2,8,bias=False)
        self.Matrix2 = nn.Linear(8,1,bias=False)
        self.R = nn.ReLU()
    def forward(self, x):
        x = self.R(self.Matrix(x))
        x = self.Matrix2(x)
        return x.squeeze()