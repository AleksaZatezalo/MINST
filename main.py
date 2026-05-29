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

# Neural Network
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(2,80)
        self.Matrix2 = nn.Linear(80,1)
        self.R = nn.ReLU()
    def forward(self, x):
        x = self.R(self.Matrix1(x))
        x = self.Matrix2(x)
        return x.squeeze()

def train_model(x,y,f,n_epochs=50):
    opt = optim.SGD(f.parameters(), lr=0.001)
    L = nn.MSELoss()

    #Train model
    losses = []
    for _ in range(n_epochs):
        opt.zero_grad() # Flush previous eopch's gradient
        loss_value = L(f(x), y)
        loss_value.backward() # Compute gradient
        opt.step()
        losses.append(loss_value.item())
    return f, losses

# Testing the neural network
x = torch.tensor(([6,2], [5,2],[1,3],[7,6])).float()
y = torch.tensor([1,5,2,5]).float()
f = MyNeuralNet()

# Train model
f2, losses2 = train_model(x,y,f, n_epochs=5000)

