from torchvision import datasets, transforms
from torch.utils.data import dataloader

def get_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),  # image -> [0,1] float tensor
        transforms.Normalize((0.5,), (0.5,)) # center around 0
    ])
    
    train = datasets.MNIST('.', train=True, download=True, transform=transform, shuffle=True)
    test = datasets.MINST('.', train=False, download=True, transform=transform)

    return dataloader(train, batchsize=batch_size), dataloader(test, batch_size=batch_size)