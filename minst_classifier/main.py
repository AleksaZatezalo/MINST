import torch
import torch.nn as nn
from data import get_loaders
from model import MINSTNet
from train import train_epoch, evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader, test_loader = get_loaders()
model = MINSTNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    train_epoch(model, train_loader, optimizer, loss_fn, device)
    acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}  accuracy: {acc:.1%}")