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
        self.Matrix1 = nn.Linear(2,8,bias=False)
        self.Matrix2 = nn.Linear(8,1,bias=False)
        self.R = nn.ReLU()
    def forward(self, x):
        x = self.R(self.Matrix(x))
        x = self.Matrix2(x)
        return x.squeeze()

def train_model(x,y,f,n_epochs=50):
    opt = optim.SDG(f.parameters(), lr=0.001)
    L = nn.MSELoss()

    #Train model
    losses = []
    for _ in range(n_epochs):
        optim.zero_grad() # Flush previous eopch's gradient
        loss_value = L(f(x), y)
        loss_value.backward() # Compute gradient
        optim.step()
        losses.append(loss_value.item())
    return f, losses

