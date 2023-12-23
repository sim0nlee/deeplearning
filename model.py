import torch
from torch import nn

import hyperparameters as hyps
from activation import TReLU


class MNIST_MLP(nn.Module):
    def __init__(self,
                 depth,
                 width,
                 activation,
                 alpha_init=1.0,
                 device="cpu",
                 trelu_is_trainable=False,
                 normalize=False):
        super().__init__()

        print(f"Network Depth: {depth}, Network Width: {width}.")
        print(f"Using {activation} activation")
        if activation == "trelu":
            print(f"TReLU alpha initialization: {alpha_init}.")
            if trelu_is_trainable:
                print("TReLU is trainable.")
        if normalize:
            print("Batch normalization on")
        print()

        self.device = device

        layers = []

        for d in range(depth):
            if d == 0:  # Input layer
                layers.append(nn.Linear(28 * 28, width))
            elif d == depth - 1:  # Last layer
                layers.append(nn.Linear(width, 10))
            else:  # Hidden layers
                layers.append(nn.Linear(width, width))
                if normalize:
                    layers.append(nn.BatchNorm1d(width))
            if d < depth - 1:  # Activation functions after all layers but the last
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "trelu":
                    layers.append(TReLU(alpha_init, trainable=trelu_is_trainable, device=self.device))

        self.net = nn.Sequential(*layers)

    def base_params(self):
        params = []
        for l in self.net:
            if isinstance(l, nn.Linear) or isinstance(l, nn.BatchNorm1d):
                for param in l.parameters():
                    params.append(param)
        return params

    def trelu_params(self):
        params = []
        for l in self.net:
            if isinstance(l, TReLU):
                for param in l.parameters():
                    params.append(param)
        return params

    def forward(self, x):
        x = nn.Flatten()(x)
        for module in self.net:
            x = x.type(torch.float32)
            x = module(x)
        return x
