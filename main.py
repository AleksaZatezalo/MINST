"""
Description: An implementation of the MINST classifcation problem in Pytorch.
Author: Aleksa Zatezalo
Date: May 2026
"""

import os
import cv2
import numpy as np
import tuple as tuple
import requests
import gzip
import hashlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.summary as summary