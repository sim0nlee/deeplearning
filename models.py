import torch
import numpy as np

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

    def forward(self, x, shallow=False):
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


class HiddenBlockCNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation_module,
                 residual=True,
                 activation_before_residual=False,
                 normalize=False):
        super().__init__()

        self.residual = residual
        self.activation_before_residual = activation_before_residual

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = activation_module
        self.bn = None
        if normalize:
            self.bn = torch.nn.BatchNorm2d(out_channels)

        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def base_params(self):
        params = []
        net = [self.conv, self.bn] if self.bn is not None else [self.conv]
        for l in net:
            for param in l.parameters():
                params.append(param)
        return params

    def trelu_params(self):
        params = []
        if isinstance(self.activation, TReLU):
            for param in self.activation.parameters():
                params.append(param)
        return params

    def forward(self, x, beta=1.0, depth_scaler=1.0):  # beta and depth_scaler are only used for residual connections
        if self.residual:
            if self.activation_before_residual:
                x_aux = beta * self.conv(x) / depth_scaler
                if self.bn is not None:
                    x_aux = self.bn(x_aux)
                x_aux = self.activation(x_aux)
                x = x + x_aux
                # x_aux = self.pool(x_aux)
            else:
                x = x + beta * self.conv(x) / depth_scaler
                if self.bn is not None:
                    x = self.bn(x)
                x = self.activation(x)
                # x = self.pool(x)

        else:
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            x = self.activation(x)
            # x = self.pool(x)

        return x


class MNIST_CNN(torch.nn.Module):
    def __init__(self,
                 depth,
                 kernels,
                 activation,
                 alpha_init=1.0,
                 device="cpu",
                 trelu_is_trainable=False,
                 residual_connections=False,
                 beta_init=1.0,
                 beta_is_trainable=False,
                 beta_is_global=True,
                 activation_before_residual=False,
                 normalize=False):
        super().__init__()

        self.depth = depth
        self.kernels = kernels
        self.activation = activation
        self.device = device
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.trelu_is_trainable = trelu_is_trainable
        self.beta_is_global = beta_is_global
        self.beta_is_trainable = beta_is_trainable
        self.residual_connections = residual_connections
        self.activation_before_residual = activation_before_residual
        self.normalize = normalize

        layers = []

        self.beta = None
        if residual_connections:
            if beta_is_global:
                self.beta = torch.nn.Parameter(torch.tensor(beta_init), requires_grad=beta_is_trainable)
            else:
                self.beta = torch.nn.ParameterList(
                    [torch.nn.Parameter(torch.tensor(beta_init), requires_grad=beta_is_trainable) for _ in
                     range(depth - 2)])

        for d in range(depth):
            if d == 0:  # Input layer
                layers.append(torch.nn.Conv2d(1, kernels, kernel_size=3, padding=1))
                if activation == "relu":  # Activation functions after all layers but the last
                    layers.append(torch.nn.ReLU())
                elif activation == "trelu":
                    layers.append(TReLU(alpha_init, trainable=trelu_is_trainable, device=self.device))
                # layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            elif d == depth - 1:  # Last layer
                layers.append(torch.nn.Conv2d(kernels, 1, kernel_size=3, padding=1))
                if activation == "relu":  # Activation functions after all layers but the last
                    layers.append(torch.nn.ReLU())
                elif activation == "trelu":
                    layers.append(TReLU(alpha_init, trainable=trelu_is_trainable, device=self.device))
                layers.append(torch.nn.Flatten())
                # layers.append(nn.Linear(32*32, 10))
                layers.append(torch.nn.Linear(28 * 28, 10))

            else:  # Hidden layers
                activation_module = torch.nn.ReLU() if activation == "relu" else TReLU(alpha_init,
                                                                                       trainable=trelu_is_trainable,
                                                                                       device=self.device)
                layers.append(HiddenBlockCNN(kernels,
                                             kernels,
                                             activation_module,
                                             residual=residual_connections,
                                             activation_before_residual=activation_before_residual,
                                             normalize=normalize))

        self.net = torch.nn.ModuleList(layers)

        self.print_model_info()

    def base_params(self):
        params = []
        for l in self.net:
            if isinstance(l, HiddenBlockCNN):
                params += l.base_params()
            elif isinstance(l, torch.nn.Linear) or isinstance(l, torch.nn.BatchNorm1d):
                for param in l.parameters():
                    params.append(param)
        return params

    def trelu_params(self):
        params = []
        for l in self.net:
            if isinstance(l, HiddenBlockCNN):
                params += l.trelu_params()
            elif isinstance(l, TReLU):
                for param in l.parameters():
                    params.append(param)
        return params

    def forward(self, x, shallow=False):
        beta_idx = 0

        if self.residual_connections and not self.beta_is_global and shallow:
            betas_tensor = torch.cat([b.abs().view(-1) for b in self.beta])
            beta_threshold = 0.1 * torch.max(betas_tensor)
            n_skipped_layers = torch.sum(betas_tensor <= beta_threshold)
            print(f"Shallow mode: Skipping {n_skipped_layers} layers with beta <= {beta_threshold}.")

        for module in self.net:
            if self.residual_connections and isinstance(module, HiddenBlockCNN):
                beta = self.beta if self.beta_is_global else self.beta[beta_idx]
                beta_idx += 1
                if self.beta_is_global or not shallow or beta.abs() > beta_threshold:
                    x = module(x, beta=beta, depth_scaler=np.sqrt(self.depth))

            else:
                x = module(x)

        return x

    def print_model_info(self):
        print(f"Network Depth: {self.depth}, Kernels: {self.kernels}.")
        print(f"Using {self.activation} activation.")
        if self.activation == "trelu":
            print(f"TReLU alpha initialization: {self.alpha_init}.")
            if self.trelu_is_trainable:
                print("TReLU is trainable.")
        if self.residual_connections:
            print("Residual connections on.")
            print(f"Residual connections beta initialization: {self.beta_init}.")
            if self.beta_is_trainable:
                print("Beta is trainable.")
                if self.beta_is_global:
                    print("Beta is global.")
                else:
                    print("Beta is per layer.")
        if self.normalize:
            print("Batch normalization on.")
        print()
