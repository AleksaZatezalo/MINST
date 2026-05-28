"""
Description: An implementation of the MINST classifcation problem in Pytorch.
Author: Aleksa Zatezalo
Date: May 2026
"""

import torch
import torch.nn as nn
import torch.optim as SGD
import numpy as np
import matplotlib.pyplot as plt

# Dummy Data
x = torch.tensor(([6,2], [5,2], [1,3],[6,7])).float()
y = torch.tensor([1,5,2,5]).float()

print(x)
print(x)