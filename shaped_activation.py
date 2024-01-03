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

hyps = {"depth"      : [150],
        "width"      : 150,
        "batch_size" : 256,
        "lr"         : 1e-2,
        "epochs"     : 6}

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=hyps["batch_size"])
test_dataloader = DataLoader(test_data, batch_size=hyps["batch_size"])

n_total_steps = len(train_dataloader)

device = "cuda" if torch.cuda.is_available() else "cpu"

examples = next(iter(test_dataloader))
example_data, example_targets = examples[0], examples[1]

class ShapedReLU_Trainable(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_max = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True))
        self.c_min = torch.nn.Parameter(torch.tensor([-10.0], requires_grad=True))
    def forward(self, x):
        s_max = 1 + self.c_max / np.sqrt(hyps["width"])
        s_min = 1 + self.c_min / np.sqrt(hyps["width"])
        return (s_max * torch.maximum(x, torch.tensor([0], device=device))
                + s_min * torch.minimum(x, torch.tensor([0], device=device)))


class ShapedReLU(nn.Module):
    def __init__(self):
        super().__init__()
        c_max = 0
        c_min = -1
        self.s_max = 1 + c_max / np.sqrt(hyps["width"])
        self.s_min = 1 + c_min / np.sqrt(hyps["width"])

    def forward(self, x):
        return (self.s_max * torch.maximum(x, torch.tensor([0], device=device))
                + self.s_min * torch.minimum(x, torch.tensor([0], device=device)))


class MLP(nn.Module):
    def __init__(self, use_batch_norm=False, use_shapedReLU=False, use_shapedReLU_trainable=False, normalizing_constant=2.0):
        super().__init__()

        layers = []

        for d in range(hyps["depth"]):
            if d == 0:
                linear_layer = nn.Linear(28 * 28, hyps["width"], bias=False)
                nn.init.normal_(linear_layer.weight.data, mean=0, std=1)
                # Scale the weights of the linear layer
                linear_layer.weight.data *= (1 / np.sqrt(28 * 28))
                layers.append(linear_layer)
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hyps["width"]))
            elif d == hyps["depth"] - 1:
                linear_layer = nn.Linear(hyps["width"], 10, bias=False)
                nn.init.normal_(linear_layer.weight.data, mean=0, std=1)
                # Scale the weights of the linear layer
                linear_layer.weight.data *= np.sqrt(normalizing_constant / hyps["width"])
                layers.append(linear_layer)
            else:
                linear_layer = nn.Linear(hyps["width"], hyps["width"], bias=False)
                nn.init.normal_(linear_layer.weight.data, mean=0, std=1)
                # Scale the weights of the linear layer
                linear_layer.weight.data *= np.sqrt(normalizing_constant / hyps["width"])
                layers.append(linear_layer)
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hyps["width"]))
            if d < hyps["depth"] - 1:
                if use_shapedReLU:
                    layers.append(ShapedReLU())
                elif use_shapedReLU_trainable:
                    layers.append(ShapedReLU_Trainable())
                else:
                    layers.append(nn.ReLU())

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
            # bias_grads = []
            for module in model.net:
                if isinstance(module, nn.Linear):
                    weight_grads.append(module.weight.grad.ravel())
                    # bias_grads.append(module.bias.grad.ravel())
                if isinstance(module, ShapedReLU_Trainable):
                    writer.add_scalar('Cmax', list(module.parameters())[0], epoch * n_total_steps + i)
                    writer.add_scalar('Cmin', list(module.parameters())[1], epoch * n_total_steps + i)
            weight_grad_norms = torch.linalg.norm(torch.cat(weight_grads))
            # bias_grad_norms = torch.linalg.norm(torch.cat(bias_grads))
            writer.add_scalar('Weight grads norm', weight_grad_norms, epoch * n_total_steps + i)
            # writer.add_scalar('Bias grads norm', bias_grad_norms, epoch * n_total_steps + i)
            print()
            print('Weight grads norm:', weight_grad_norms)
            # print('Bias grads norm:', bias_grad_norms)
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

    model = MLP(use_batch_norm=False, use_shapedReLU=False).to(device)
    # model_bn = MLP(use_batch_norm=True, use_shapedReLU=False).to(device)
    model_sReLU = MLP(use_batch_norm=False, use_shapedReLU=True).to(device)
    # model_sReLU_train = MLP(use_batch_norm=False, use_shapedReLU=False, use_shapedReLU_trainable=True).to(device)

    writer = SummaryWriter(f"runs/mnist/without_bn/with_ReLU/depth_{depth}")
    # writer_bn = SummaryWriter(f"runs/mnist/with_bn/with_ReLU/depth_{depth}")
    writer_sReLU = SummaryWriter(f"runs/mnist/without_bn/with_shapedReLu/depth_{depth}")
    # writer_sReLU_train = SummaryWriter(f"runs/mnist/without_bn/with_shapedReLu_train/depth_{depth}")

    for t in range(hyps["epochs"]):
        optimizer = torch.optim.Adam(model.parameters(), lr=hyps["lr"])

        writer.add_graph(model, example_data)
        # writer_bn.add_graph(model_bn, example_data)
        writer_sReLU.add_graph(model_sReLU, example_data)
        #writer_sReLU_train.add_graph(model_sReLU_train, example_data)

        print(f"Depth: {depth}, Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, criterion, optimizer, t, writer=writer)
        test(test_dataloader, model, criterion)

        # train(train_dataloader, model_bn, criterion, optimizer_bn, t, writer=writer_bn)
        # test(test_dataloader, model_bn, criterion)

        train(train_dataloader, model_sReLU, criterion, optimizer, t, writer=writer_sReLU)
        test(test_dataloader, model_sReLU, criterion)

        # train(train_dataloader, model_sReLU_train, criterion, optimizer_sReLu_train, t, writer=writer_sReLU_train)
        # test(test_dataloader, model_sReLU_train, criterion)

    writer.close()
    #writer_bn.close()
    writer_sReLU.close()
    #writer_sReLU_train.close()


print("Done!")