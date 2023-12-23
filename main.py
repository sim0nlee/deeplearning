import torch

from data import train_dataloader, test_dataloader
from train import train
from test import test
from model import MNIST_MLP
from activation import optimal_trelu_params

import hyperparameters as hyps

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

device = "cuda" if torch.cuda.is_available() else "cpu"

activation = "trelu"  # "relu"

OPTIMAL_ALPHA_1, OPTIMAL_ALPHA_2 = list(optimal_trelu_params())
alpha = 1.0

model = MNIST_MLP(hyps.depth,
                  hyps.width,
                  activation,
                  alpha,
                  device,
                  trelu_is_trainable=True,
                  normalize=True).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params': model.base_params()},
    {'params': model.trelu_params(), 'lr': hyps.adam_alpha_lr}
])

if __name__ == "__main__":
    for epoch in range(hyps.epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, criterion, optimizer, epoch, device, writer)
        test(test_dataloader, model, criterion, epoch, device, writer)
