import torch
import torch.distributed as dist
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader, DistributedSampler

def get_dataloaders(batch_size=64, download_dir="data/"):

    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if dist.get_rank() == 0:
        datasets.CIFAR10(download_dir, download=True, train=True, transform=transform)
        datasets.CIFAR10(download_dir, download=True, train=False, transform=transform)
    dist.barrier()

    trainset = datasets.CIFAR10(download_dir, train=True, transform=transform)
    testset = datasets.CIFAR10(download_dir, train=False, transform=transform)

    train_sampler = DistributedSampler(trainset, shuffle=True)
    test_sampler = DistributedSampler(testset, shuffle=False)

    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
    testloader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler)

    return trainloader, testloader
