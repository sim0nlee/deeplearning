import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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

hyps = {"depth"      : 6,
        "width"      : 100,
        "batch_size" : 256,
        "lr"         : 1e0,
        "epochs"     : 10,
        "s_max"      : 0.1,
        "s_min"      : 0.05}

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=hyps["batch_size"])
test_dataloader = DataLoader(test_data, batch_size=hyps["batch_size"])

device = "cuda" if torch.cuda.is_available() else "cpu"


class ShapedReLU(nn.Module):
    def __init__(self, s_max, s_min):
        super().__init__()
        self.s_max = s_max
        self.s_min = s_min

    def forward(self, x):
        return self.s_max * torch.maximum(x, torch.tensor([0])) + self.s_min * torch.minimum(x, torch.tensor([0]))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []

        for d in range(hyps["depth"]):
            if d == 0:
                layers.append(nn.Linear(28 * 28, hyps["width"]))
            elif d == hyps["depth"] - 1:
                layers.append(nn.Linear(hyps["width"], 10))
            else:
                layers.append(nn.Linear(hyps["width"], hyps["width"]))
            if d < hyps["depth"] - 1:
                layers.append(ShapedReLU(hyps["s_max"], hyps["s_min"]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = nn.Flatten()(x)
        logits = self.net(x)
        return logits


model = MLP().to(device)

x = torch.linspace(-10, 10, 1000)
plt.plot(x, ShapedReLU(hyps["s_max"], hyps["s_min"])(x))
plt.show()

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams["lr"])


# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for i, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)
#
#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)
#
#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#
#         if i % 100 == 0:
#             weight_grads = []
#             bias_grads = []
#             for module in model.net:
#                 if isinstance(module, nn.Linear):
#                     weight_grads.append(module.weight.grad.ravel())
#                     bias_grads.append(module.bias.grad.ravel())
#             weight_grad_norms = torch.linalg.norm(torch.cat(weight_grads))
#             bias_grad_norms = torch.linalg.norm(torch.cat(bias_grads))
#             print()
#             print('Weight grads norm:', weight_grad_norms)
#             print('Bias grads norm:', bias_grad_norms)
#             print()
#
#         optimizer.zero_grad()
#
#         if i % 25 == 0:
#             loss, current = loss.item(), (i + 1) * len(X)
#             print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#
#
# def test(dataloader, model, loss_fn):
#     size        = len(dataloader.dataset)
#     num_batches = len(dataloader)
#
#     model.eval()
#
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#
#
# for t in range(hyperparams["epochs"]):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, criterion, optimizer)
#     test(test_dataloader, model, criterion)
# print("Done!")