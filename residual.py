import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np

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

hyperparams = {"batch_size": 256,
               "lr"        : 1e0,
               "epochs"    : 10,
               "depth"     : 5,
               "width"     : 100}


BETA_init = 1.0

train_dataloader = DataLoader(training_data, batch_size=hyperparams["batch_size"])
test_dataloader = DataLoader(test_data, batch_size=hyperparams["batch_size"])

device = "cuda" if torch.cuda.is_available() else "cpu"


class Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor(BETA_init, dtype=torch.float32), requires_grad=True)
        self.L = hyperparams["depth"]
        self.linear = nn.Linear(hyperparams["width"], hyperparams["width"])
        # weights = torch.Tensor(size_out, size_in)
        # bias = torch.Tensor(size_out)
        # self.weights = torch.nn.Parameter(weights, requires_grad=True)
        # self.bias = torch.nn.Parameter(bias, requires_grad=True)

    def forward(self, x):
        return self.beta/np.sqrt(self.L) * self.linear(x) + x
        #return self.beta/np.sqrt(self.L) * torch.mm(x, self.weights.t()) + x
    

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []

        for d in range(hyperparams["depth"]):
            if d == 0:
                layers.append(nn.Linear(28 * 28, hyperparams["width"]))
            elif d == hyperparams["depth"] - 1:
                layers.append(nn.Linear(hyperparams["width"], 10))
            else:
                layers.append(nn.ReLU())
                layers.append(Residual())
                
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = nn.Flatten()(x)
        logits = self.net(x)
        return logits


model = MLP().to(device)
print(model)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams["lr"])


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


for t in range(hyperparams["epochs"]):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, criterion, optimizer)
    test(test_dataloader, model, criterion)
print("Done!")

