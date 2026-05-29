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

x = torch.tensor(([4,6,2,-1,6,2,5],[1,6,2,-6,5,-3,5]))
print(x)

R = nn.ReLU()
print(R(x))