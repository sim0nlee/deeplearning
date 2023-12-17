import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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

hyps = {"depth"      : 20,
        "width"      : 100,
        "batch_size" : 256,
        "lr"         : 1e0,
        "epochs"     : 10}

train_dataloader = DataLoader(training_data, batch_size=hyps["batch_size"], shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=hyps["batch_size"])

device = "cuda" if torch.cuda.is_available() else "cpu"


class TReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x):
        return torch.sqrt(2. / (1. + self.alpha ** 2.)) * \
                (torch.maximum(x, torch.tensor([0], device=device)) +
                self.alpha * torch.minimum(x, torch.tensor([0], device=device)))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []

        for d in range(hyps["depth"]):
            if d == 0:  # Input layer
                layers.append(nn.Linear(28 * 28, hyps["width"]))
            elif d == hyps["depth"] - 1:  # Last layer
                layers.append(nn.Linear(hyps["width"], 10))
            else:  # Hidden layers
                layers.append(nn.Linear(hyps["width"], hyps["width"]))
            if d < hyps["depth"] - 1:  # Activation functions after all layers but the last
                layers.append(TReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = nn.Flatten()(x)
        logits = self.net(x)
        return logits


model = MLP().to(device)
# print(model)

# x = torch.linspace(-10, 10, 1000)
# plt.plot(x, ShapedReLU(hyps["s_max"], hyps["s_min"])(x))
# plt.show()

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=hyps["lr"])
optimizer = torch.optim.Adam(model.parameters())

def train(dataloader, model, loss_fn, optimizer):
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

        if i % 100 == 0:
            weight_grads = []
            bias_grads = []
            alphas = []
            for module in model.net:
                if isinstance(module, nn.Linear):
                    weight_grads.append(module.weight.grad.ravel())
                    bias_grads.append(module.bias.grad.ravel())
                if isinstance(module, TReLU):
                    for param in module.parameters():
                        alphas.append(param.data.item())

            weight_grad_norms = torch.linalg.norm(torch.cat(weight_grads))
            bias_grad_norms = torch.linalg.norm(torch.cat(bias_grads))
            print()
            print('Weight grads norm:', weight_grad_norms.item())
            print('Bias grads norm:', bias_grad_norms.item())
            print('TReLU alphas:', ["{:.5f}".format(alpha) for alpha in alphas])
            print()

        optimizer.zero_grad()

        if i % 25 == 0:
            loss, current = loss.item(), (i + 1) * len(X)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n\n")


for t in range(hyps["epochs"]):
    print(f"EPOCH {t+1}\n-------------------------------")
    train(train_dataloader, model, criterion, optimizer)
    test(test_dataloader, model, criterion)
print("Done!")

