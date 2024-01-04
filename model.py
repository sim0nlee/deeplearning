import torch

from activation import TReLU


class MNIST_MLP(torch.nn.Module):
    def __init__(self,
                 device,
                 depth,
                 width,
                 activation,
                 alpha_init=1.0,
                 trelu_is_trainable=False,
                 normalize=False):
        super().__init__()

        if activation not in ["relu", "lrelu", "trelu"]:
            raise Exception("Invalid Activation")

        self.device = device
        self.depth = depth
        self.width = width
        self.activation = activation
        self.alpha_init = alpha_init
        self.trelu_is_trainable = trelu_is_trainable
        self.normalize = normalize

        layers = []

        for d in range(depth):
            # Input layer
            if d == 0:
                layers.append(torch.nn.Linear(28 * 28, width))
            # Last layer
            elif d == depth - 1:
                layers.append(torch.nn.Linear(width, 10))
            # Hidden layers
            else:
                layer = torch.nn.Linear(width, width)
                # Custom weight and bias initialization
                torch.nn.init.normal_(layer.weight, 0, 1 / width)
                torch.nn.init.constant_(layer.bias, 0)
                layers.append(torch.nn.Linear(width, width))
                if normalize:
                    layers.append(torch.nn.BatchNorm1d(width))
            # Activation functions after all layers but the last
            if d < depth - 1:
                if activation == "relu":
                    layers.append(torch.nn.ReLU())
                elif activation == "lrelu":
                    layers.append(torch.nn.LeakyReLU())
                elif activation == "trelu":
                    layers.append(TReLU(alpha_init, trainable=trelu_is_trainable, device=self.device))

        self.net = torch.nn.Sequential(*layers)

        self.print_model_info()

    def base_params(self):
        """Returns a list of the linear layer and batch normalization layer parameters"""
        params = []
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.BatchNorm1d):
                for param in layer.parameters():
                    params.append(param)
        return params

    def trelu_params(self):
        """Returns a list of the alpha parameters of each TReLU layer of the network, if present"""
        params = []
        for layer in self.net:
            if isinstance(layer, TReLU):
                for param in layer.parameters():
                    params.append(param)
        return params

    def forward(self, x):
        x = torch.nn.Flatten()(x)
        for module in self.net:
            x = x.type(torch.float32)
            x = module(x)
        return x

    def print_model_info(self):
        print(f"Network Depth: {self.depth}, Network Width: {self.width}.")
        print(f"Using {self.activation} activation")
        if self.activation == "trelu":
            print(f"TReLU alpha initialization: {self.alpha_init}.")
            if self.trelu_is_trainable:
                print("TReLU is trainable.")
            else:
                print("TReLU is not trainable")
        if self.normalize:
            print("Batch normalization on")
        else:
            print("Batch normalization off")
        print()
