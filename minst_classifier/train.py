import torch

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    for img, lables in loader:
        img, lables = img.to(device), lables.to(device)
        optimizer.zero_grad()   # Clear old gradients
        loss = loss_fn(model(img), lables)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, lables in loader:
            images. lables = images.to(device), lables.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds==lables).sum().item()
            total += lables.size(0)
    return correct / total
