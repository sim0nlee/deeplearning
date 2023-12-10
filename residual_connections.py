import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.nn import functional as F

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

hyps = {"depth"      : 5,
        "width"      : 100,
        "batch_size" : 256,
        "lr"         : 1e0,
        "epochs"     : 5}


beta_ini = 1.0
trainable_beta = True
global_beta = True

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=hyps["batch_size"], shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=hyps["batch_size"])

device = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []

        if global_beta:
            self.beta = nn.Parameter(torch.tensor(beta_ini), requires_grad=trainable_beta)
        else:
            self.beta = nn.ParameterList([nn.Parameter(torch.tensor(beta_ini), requires_grad=trainable_beta) for _ in range(hyps["depth"]-2)])

        for d in range(hyps["depth"]):
            if d == 0:
                layers.append(nn.Linear(28 * 28, hyps["width"]))
            elif d == hyps["depth"] - 1:
                layers.append(nn.Linear(hyps["width"], 10))
            else:
                layers.append(nn.Linear(hyps["width"], hyps["width"]))

        self.net = nn.ModuleList(layers)

    def forward(self, x):
        x = nn.Flatten()(x)

        x = self.net[0](x)
        x = F.relu(x)

        for i, layer in enumerate(self.net[1:-1]):
            beta = self.beta if global_beta else self.beta[i]
            x = x + beta * layer(x) / np.sqrt(hyps["depth"]-2) # TODO: depth or depth-2?
            x = F.relu(x) # TODO: ReLU after or before residual connection?

        logits = self.net[-1](x)
        return logits


model = MLP().to(device)
print(model)

# x = torch.linspace(-10, 10, 1000)
# plt.plot(x, ShapedReLU(hyps["s_max"], hyps["s_min"])(x))
# plt.show()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=hyps["lr"])


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
            for module in model.net:
                if isinstance(module, nn.Linear):
                    weight_grads.append(module.weight.grad.ravel())
                    bias_grads.append(module.bias.grad.ravel())
            weight_grad_norms = torch.linalg.norm(torch.cat(weight_grads))
            bias_grad_norms = torch.linalg.norm(torch.cat(bias_grads))
            print()
            print('Weight grads norm:', weight_grad_norms)
            print('Bias grads norm:', bias_grad_norms)
            if global_beta:
                print('Global Beta:', model.beta.item())
            else:
                print('Betas:', [b.item() for b in model.beta])
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(hyps["epochs"]):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, criterion, optimizer)
    test(test_dataloader, model, criterion)
print("Done!")
