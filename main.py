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

# Scaling
Image = torch.randn((256,256,3))
Scale = torch.Tensor([0.5, 1.5, 1])
print(Scale)
result = Image * Scale
print(Image)

# Operations across Dimensions
t = torch.tensor([0.5, 1, 3, 4])
torch.mean(t), torch.std(t), torch.max(t), torch.min(t)
t = torch.arange(20, dtype=float).reshape(5,4)
print(torch.mean(t, axis=0))