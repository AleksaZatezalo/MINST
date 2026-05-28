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

x = torch.tensor(([6,2], [5,2], [1,3],[6,7])).float()
y = torch.tensor([1,5,2,5]).float()

class MyNeuralNet(nn.Module):
    """Basic class that inherits from nn.Module"""

    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(2, 8, bias=False)
        self.Matrix2 = nn.Linear(8, 1, bias=False)
    def forward(self, x):
        x = self.Matrix1(x)
        x = self.Matrix2(x)
        return x.squeeze()

f = MyNeuralNet()

# Setting up optimizer
optimizer = optim.SGD(f.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
losses = []

# Training Data
for epoch in range(100):
    optimizer.zero_grad()
    output = f(x)
    loss_value = loss_fn(output, y)
    
    loss_value.backward()
    optimizer.step()
    
    losses.append(loss_value.item())
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss_value.item():.6f}")

# Creating plot of loss
plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.grid(True)
plt.show()