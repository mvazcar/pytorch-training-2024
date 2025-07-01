import torch
import torch.optim as optim
import torch.nn as nn
#import torch.multiprocessing as mp
from model import CNN
from train import train
from data import get_dataloaders
from distributed_utils import setup, cleanup
#from fsdp_utils import get_fsdp_model
#from tensor_parallel import tensor_parallel_split
import argparse

def run(args):
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    trainloader, validloader = get_dataloaders()

    #if args.method == "ddp":
        model = CNN().to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    #elif args.method == "fsdp":
    #    model = get_fsdp_model()
    #elif args.method == "tensor":
    #    model = CNN()
    #    model = tensor_parallel_split(model)  # Warning: illustrative only
    #else:
    #    raise ValueError(f"Unknown method: {args.method}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.NLLLoss()

    train(model, optimizer, loss_function, args.epochs, device, trainloader, validloader)

    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["ddp", "fsdp", "tensor"])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()
