import torch
import torchvision.utils
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor

import sys

import numpy as np

from matplotlib import pyplot as plt

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

hyps = {"depth"      : [10, 20, 50, 100],
        "width"      : 100,
        "batch_size" : 256,
        "lr"         : 1e0,
        "epochs"     : 6}

c_max = 0
c_min = -10
s_max = 1 + c_max / np.sqrt(hyps["width"])
s_min = 1 + c_min / np.sqrt(hyps["width"])

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=hyps["batch_size"])
test_dataloader = DataLoader(test_data, batch_size=hyps["batch_size"])

n_total_steps = len(train_dataloader)

device = "cuda" if torch.cuda.is_available() else "cpu"

examples = next(iter(test_dataloader))
example_data, example_targets = examples[0], examples[1]


class ShapedReLU(nn.Module):
    def __init__(self, s_max, s_min):
        super().__init__()
        self.s_max = s_max
        self.s_min = s_min

    def forward(self, x):
        return (self.s_max * torch.maximum(x, torch.tensor([0], device=device))
                + self.s_min * torch.minimum(x, torch.tensor([0], device=device)))


class MLP(nn.Module):
    def __init__(self, use_batch_norm=False):
        super().__init__()

        layers = []

        for d in range(hyps["depth"]):
            if d == 0:
                layers.append(nn.Linear(28 * 28, hyps["width"]))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hyps["width"]))
            elif d == hyps["depth"] - 1:
                layers.append(nn.Linear(hyps["width"], 10))
            else:
                layers.append(nn.Linear(hyps["width"], hyps["width"]))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hyps["width"]))
            if d < hyps["depth"] - 1:
                layers.append(ShapedReLU(s_max, s_min))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = nn.Flatten()(x)
        logits = self.net(x)
        return logits

def train(dataloader, model, loss_fn, optimizer, epoch, writer=None):
    size = len(dataloader.dataset)
    model.train()
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if writer is not None and i % 100 == 0:
            weight_grads = []
            bias_grads = []
            for module in model.net:
                if isinstance(module, nn.Linear):
                    weight_grads.append(module.weight.grad.ravel())
                    bias_grads.append(module.bias.grad.ravel())
            weight_grad_norms = torch.linalg.norm(torch.cat(weight_grads))
            bias_grad_norms = torch.linalg.norm(torch.cat(bias_grads))
            writer.add_scalar('Weight grads norm', weight_grad_norms, epoch * n_total_steps + i)
            writer.add_scalar('Bias grads norm', bias_grad_norms, epoch * n_total_steps + i)
            print()
            print('Weight grads norm:', weight_grad_norms)
            print('Bias grads norm:', bias_grad_norms)
            print()

        optimizer.zero_grad()

        if i % 25 == 0:
            loss, current = loss.item(), (i + 1) * len(X)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            writer.add_scalar('training loss', loss, epoch * n_total_steps + i)


def test(dataloader, model, loss_fn):
    size        = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


criterion = nn.CrossEntropyLoss()

for depth in hyps["depth"]:
    hyps["depth"] = depth

    model_without_bn = MLP(use_batch_norm=False).to(device)
    model_with_bn = MLP(use_batch_norm=True).to(device)

    writer_without_bn = SummaryWriter(f"runs/mnist/without_bn/depth_{depth}")
    writer_with_bn = SummaryWriter(f"runs/mnist/with_bn/depth_{depth}")

    for t in range(hyps["epochs"]):
        optimizer_without_bn = torch.optim.SGD(model_without_bn.parameters(), lr=hyps["lr"])
        optimizer_with_bn = torch.optim.SGD(model_with_bn.parameters(), lr=hyps["lr"])

        writer_without_bn.add_graph(model_without_bn, example_data)
        writer_with_bn.add_graph(model_with_bn, example_data)

        print(f"Depth: {depth}, Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model_without_bn, criterion, optimizer_without_bn, t, writer=writer_without_bn)
        test(test_dataloader, model_without_bn, criterion)

        train(train_dataloader, model_with_bn, criterion, optimizer_with_bn, t, writer=writer_with_bn)
        test(test_dataloader, model_with_bn, criterion)

    writer_without_bn.close()
    writer_with_bn.close()



print("Done!")