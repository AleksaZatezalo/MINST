import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data  import get_loaders
from model import MINSTNet
from train import train_epoch, evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader, test_loader = get_loaders()
model     = MINSTNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.CrossEntropyLoss()

for epoch in range(5):
    train_epoch(model, train_loader, optimizer, loss_fn, device)
    acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}  accuracy: {acc:.1%}")

def show_predictions(model, loader, device, n=10):
    model.eval()
    images, labels = next(iter(loader))     # grab one batch
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        preds = model(images).argmax(dim=1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().squeeze()     # remove channel dim: (1,28,28) -> (28,28)
        ax.imshow(img, cmap='gray')
        color = 'green' if preds[i] == labels[i] else 'red'
        ax.set_title(f"pred: {preds[i].item()}  true: {labels[i].item()}", color=color)
        ax.axis('off')

    plt.suptitle('Green = correct, Red = wrong', y=1.02)
    plt.tight_layout()
    plt.show()

show_predictions(model, test_loader, device)