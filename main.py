"""
Description: An implementation of the MINST classifcation problem in Pytorch.
Author: Aleksa Zatezalo
Date: May 2026
"""

import torch

# Creating Linspace
t = torch.linspace(0, 1, 5)
print(t)

# Creating a Tensor
t = torch.arange(48).reshape(3, 4, 4)
print(t)