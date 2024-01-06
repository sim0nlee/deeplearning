import torch
from torch import nn
import numpy as np

import hyperparameters as hyps
from activation import TReLU


class HiddenBlock(nn.Module):
    def __init__(self,
                 width,
                 activation_module,
                 residual=True,
                 activation_before_residual=False,
                 normalize=False):
        super().__init__()

        self.residual = residual
        self.activation_before_residual = activation_before_residual

        self.linear = nn.Linear(width, width)
        self.activation = activation_module
        self.bn = None
        if normalize:
            self.bn = nn.BatchNorm1d(width)

    def base_params(self):
        params = []
        net = [self.linear, self.bn] if self.bn is not None else [self.linear]
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

    def forward(self, x, beta=1.0, depth_scaler=1.0): # beta and depth_scaler are only used for residual connections
        if self.residual:
            if self.activation_before_residual:
                x_aux = beta * self.linear(x) / depth_scaler
                if self.bn is not None:
                    x_aux = self.bn(x_aux)
                x_aux = self.activation(x_aux)
                x = x + x_aux
            else:
                x = x + beta * self.linear(x) / depth_scaler 
                if self.bn is not None:
                    x_aux = self.bn(x_aux)
                x = self.activation(x)

        else:
            x = self.linear(x)
            if self.bn is not None:
                x = self.bn(x)
            x = self.activation(x)

        return x
    


class MNIST_MLP(nn.Module):
    def __init__(self,
                 depth,
                 width,
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

        print(f"Network Depth: {depth}, Network Width: {width}.")
        print(f"Using {activation} activation.")
        if activation == "trelu":
            print(f"TReLU alpha initialization: {alpha_init}.")
            if trelu_is_trainable:
                print("TReLU is trainable.")
        if residual_connections:
            print("Residual connections on.")
            print(f"Residual connections beta initialization: {beta_init}.")
            if beta_is_trainable:
                print("Beta is trainable.")
                if beta_is_global:
                    print("Beta is global.")
                else:
                    print("Beta is per layer.")
        if normalize:
            print("Batch normalization on.")
        print()

        self.device = device
        self.beta_is_global = beta_is_global
        self.beta_is_trainable = beta_is_trainable
        self.residual_connections = residual_connections
        self.depth = depth
        self.activation_before_residual = activation_before_residual

        layers = []

        self.beta = None
        if residual_connections:
            if beta_is_global:
                self.beta = nn.Parameter(torch.tensor(beta_init), requires_grad=beta_is_trainable)
            else:
                self.beta = nn.ParameterList([nn.Parameter(torch.tensor(beta_init), requires_grad=beta_is_trainable) for _ in range(depth-2)])

        for d in range(depth):
            if d == 0:  # Input layer
                #layers.append(nn.Linear(28 * 28, width))
                layers.append(nn.Linear(12288, width))
                if activation == "relu": # Activation functions after all layers but the last
                    layers.append(nn.ReLU())
                elif activation == "trelu":
                    layers.append(TReLU(alpha_init, trainable=trelu_is_trainable, device=self.device))
           
            elif d == depth - 1:  # Last layer
                #layers.append(nn.Linear(width, 10))
                layers.append(nn.Linear(width, 200))

            else:  # Hidden layers
                activation_module = nn.ReLU() if activation == "relu" else TReLU(alpha_init, trainable=trelu_is_trainable, device=self.device)
                layers.append(HiddenBlock(width,
                                          activation_module,
                                          residual=residual_connections,
                                          activation_before_residual=activation_before_residual,
                                          normalize=normalize))

        self.net = nn.ModuleList(layers)

    
    def base_params(self):
        params = []
        for l in self.net:
            if isinstance(l, HiddenBlock):
                params += l.base_params()
            elif isinstance(l, nn.Linear) or isinstance(l, nn.BatchNorm1d):
                for param in l.parameters():
                    params.append(param)
        return params
    

    def trelu_params(self):
        params = []
        for l in self.net:
            if isinstance(l, HiddenBlock):
                params += l.trelu_params()
            elif isinstance(l, TReLU):
                for param in l.parameters():
                    params.append(param)
        return params


    def forward(self, x, shallow=False):
        x = nn.Flatten()(x)

        beta_idx = 0

        if self.residual_connections and not self.beta_is_global and shallow:
            betas_tensor = torch.cat([b.view(-1) for b in self.beta])
            beta_threshold = 0.1 * torch.max(betas_tensor)
            n_skipped_layers = torch.sum(betas_tensor <= beta_threshold)
            print(f"Shallow mode: Skipping {n_skipped_layers} layers with beta <= {beta_threshold}.")

        for module in self.net:
            if self.residual_connections and isinstance(module, HiddenBlock):
                beta = self.beta if self.beta_is_global else self.beta[beta_idx]
                beta_idx += 1
                if not shallow or beta > beta_threshold:
                    x = module(x, beta=beta, depth_scaler=np.sqrt(self.depth))

            else:
                x = module(x)
        
        return x
    


# Hidden Block but using Conv2d, MaxPool instead of Linear
class HiddenBlockCNN(nn.Module):
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

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = activation_module
        self.bn = None
        if normalize:
            self.bn = nn.BatchNorm2d(out_channels)
        
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

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
    

    def forward(self, x, beta=1.0, depth_scaler=1.0): # beta and depth_scaler are only used for residual connections
        if self.residual:
            if self.activation_before_residual:
                x_aux = beta * self.conv(x) / depth_scaler
                if self.bn is not None:
                    x_aux = self.bn(x_aux)
                x_aux = self.activation(x_aux)
                x = x + x_aux
                #x_aux = self.pool(x_aux)
            else:
                x = x + beta * self.conv(x) / depth_scaler 
                if self.bn is not None:
                    x = self.bn(x)
                x = self.activation(x)
                #x = self.pool(x)

        else:
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            x = self.activation(x)
            #x = self.pool(x)

        return x
    


class IMAGENET_CNN(nn.Module):
    def __init__(self,
                 depth,
                 width,
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

        print(f"Network Depth: {depth}, Network Width: {width}.")
        print(f"Using {activation} activation.")
        if activation == "trelu":
            print(f"TReLU alpha initialization: {alpha_init}.")
            if trelu_is_trainable:
                print("TReLU is trainable.")
        if residual_connections:
            print("Residual connections on.")
            print(f"Residual connections beta initialization: {beta_init}.")
            if beta_is_trainable:
                print("Beta is trainable.")
                if beta_is_global:
                    print("Beta is global.")
                else:
                    print("Beta is per layer.")
        if normalize:
            print("Batch normalization on.")
        print()

        self.device = device
        self.beta_is_global = beta_is_global
        self.beta_is_trainable = beta_is_trainable
        self.residual_connections = residual_connections
        self.depth = depth
        self.activation_before_residual = activation_before_residual

        layers = []

        self.beta = None
        if residual_connections:
            if beta_is_global:
                self.beta = nn.Parameter(torch.tensor(beta_init), requires_grad=beta_is_trainable)
            else:
                self.beta = nn.ParameterList([nn.Parameter(torch.tensor(beta_init), requires_grad=beta_is_trainable) for _ in range(depth-2)])

        for d in range(depth):
            if d == 0:  # Input layer
                layers.append(nn.Conv2d(1, width, kernel_size=3, padding=1))
                if activation == "relu": # Activation functions after all layers but the last
                    layers.append(nn.ReLU())
                elif activation == "trelu":
                    layers.append(TReLU(alpha_init, trainable=trelu_is_trainable, device=self.device))
                #layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
           
            elif d == depth - 1:  # Last layer
                layers.append(nn.Conv2d(width, 1, kernel_size=3, padding=1))
                if activation == "relu": # Activation functions after all layers but the last
                    layers.append(nn.ReLU())
                elif activation == "trelu":
                    layers.append(TReLU(alpha_init, trainable=trelu_is_trainable, device=self.device))
                layers.append(nn.Flatten())
                #layers.append(nn.Linear(32*32, 10))
                layers.append(nn.Linear(28*28, 10))


            else:  # Hidden layers
                activation_module = nn.ReLU() if activation == "relu" else TReLU(alpha_init, trainable=trelu_is_trainable, device=self.device)
                layers.append(HiddenBlockCNN(width,
                                             width,
                                            activation_module,
                                            residual=residual_connections,
                                            activation_before_residual=activation_before_residual,
                                            normalize=normalize))

        self.net = nn.ModuleList(layers)

    
    def base_params(self):
        params = []
        for l in self.net:
            if isinstance(l, HiddenBlockCNN):
                params += l.base_params()
            elif isinstance(l, nn.Linear) or isinstance(l, nn.BatchNorm1d):
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

        for module in self.net[:-1]:
            if self.residual_connections and isinstance(module, HiddenBlockCNN):
                beta = self.beta if self.beta_is_global else self.beta[beta_idx]
                beta_idx += 1
                if self.beta_is_global or not shallow or beta.abs() > beta_threshold:
                    x = module(x, beta=beta, depth_scaler=np.sqrt(self.depth))

            else:
                x = module(x)
        
        return x
    