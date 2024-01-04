# import numpy as np
#
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor
#
# from matplotlib import pyplot as plt
# from scipy.optimize import fsolve
#
# import hyperparameters as hyps
#
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
#
# training_data = datasets.MNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )
#
# test_data = datasets.MNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )
#
# train_dataloader = DataLoader(training_data, batch_size=hyps.batch_size, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=hyps.batch_size)
#
# n_total_steps = len(train_dataloader)
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
#
# class TReLU(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
#         # self.alpha = torch.tensor([1.75303419], device=device)
#         # self.alpha = torch.tensor([0.57043953], device=device)
#
#     def forward(self, x):
#         return torch.sqrt(2. / (1. + self.alpha ** 2.)) * \
#                 (torch.maximum(x, torch.tensor([0], device=device)) +
#                 self.alpha * torch.minimum(x, torch.tensor([0], device=device)))
#
#
# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         layers = []
#
#         for d in range(hyps.depth):
#             if d == 0:  # Input layer
#                 layers.append(nn.Linear(28 * 28, hyps.width))
#             elif d == hyps.depth - 1:  # Last layer
#                 layers.append(nn.Linear(hyps.width, 10))
#             else:  # Hidden layers
#                 layers.append(nn.Linear(hyps.width, hyps.width))
#                 layers.append(nn.BatchNorm1d(hyps.width))
#             if d < hyps.depth - 1:  # Activation functions after all layers but the last
#                 # layers.append(torch.nn.ReLU())
#                 layers.append(TReLU())
#
#         self.net = nn.Sequential(*layers)
#
#     def base_params(self):
#         params = []
#         for l in self.net:
#             if isinstance(l, nn.Linear) or isinstance(l, nn.BatchNorm1d):
#                 for param in l.parameters():
#                     params.append(param)
#         return params
#
#     def trelu_params(self):
#         params = []
#         for l in self.net:
#             if isinstance(l, TReLU):
#                 for param in l.parameters():
#                     params.append(param)
#         return params
#
#     def forward(self, x):
#         x = nn.Flatten()(x)
#         logits = self.net(x)
#         return logits
#
#
# def C(c, alpha):
#     """Returns the value of the C map for phi~(x) with given alpha"""
#     return c + (((1 - alpha) ** 2) / (torch.pi * (1 + alpha ** 2))) * (np.sqrt(1 - c ** 2) - c * np.arccos(c))
#
#
# def C_f(alpha, c=0):
#     cf = C(c, alpha)
#     for _ in range(hyps.depth):
#         cf = C(cf, alpha)
#     return cf
#
#
# func = lambda x: C_f(x) - hyps.eta
#
# # alphas = list(torch.linspace(-10, 10, 1000))
# # cfs = [func(alpha) for alpha in alphas]
# # plt.plot(alphas, cfs, '-')
# # plt.show()
# #
# # print(fsolve(func, x0=np.array([0.0, 1.0])))
#
# ########################################
# ############## TRAINING ################
# ########################################
#
# model = MLP().to(device)
#
#
# criterion = nn.CrossEntropyLoss()
# #optimizer = torch.optim.SGD(model.parameters(), lr=hyps["sgdlr"])
# optimizer = torch.optim.Adam([
#     {'params': model.base_params()},
#     {'params': model.trelu_params(), 'lr': 1e-2}
# ])
# optimizer = torch.optim.Adam(model.parameters())
#
# def train(dataloader, model, loss_fn, optimizer, epoch, writer=None):
#     size = len(dataloader.dataset)
#     model.train()
#
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
#             alphas = []
#             for module in model.net:
#                 if isinstance(module, nn.Linear):
#                     weight_grads.append(module.weight.grad.ravel())
#                     bias_grads.append(module.bias.grad.ravel())
#                 if isinstance(module, TReLU):
#                     for param in module.parameters():
#                         alphas.append(param.data.item())
#
#             weight_grad_norms = torch.linalg.norm(torch.cat(weight_grads))
#             bias_grad_norms = torch.linalg.norm(torch.cat(bias_grads))
#             print()
#             print('Weight grads norm:', weight_grad_norms.item())
#             print('Bias grads norm:', bias_grad_norms.item())
#             writer.add_scalar('Weights Gradient Norms', weight_grad_norms.item(), n_total_steps * epoch + i)
#             writer.add_scalar('Bias Gradient Norms', bias_grad_norms.item(), n_total_steps * epoch + i)
#             print('TReLU alphas:', ["{:.5f}".format(alpha) for alpha in alphas])
#             print()
#
#         optimizer.zero_grad()
#
#         if i % (2**5) == 0:
#             loss, current = loss.item(), (i + 1) * len(X)
#             if writer is not None:
#                 writer.add_scalar('Training Loss', loss, n_total_steps * epoch + i)
#             print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#
#
# def test(dataloader, model, loss_fn, epoch, writer=None):
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
#     print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n\n")
#     writer.add_scalar('Test Accuracy', correct, epoch)
#
#
# for epoch in range(hyps.epochs):
#     print(f"EPOCH {epoch + 1}\n-------------------------------")
#     train(train_dataloader, model, criterion, optimizer, epoch, writer)
#     test(test_dataloader, model, criterion, epoch, writer)
# print("Done!")
#
# writer.close()
