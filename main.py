"""
The following file can be used to reproduce our experiments. By modifying the hyperparameters defined below,
different kind of experiments can be carried out. Note that setting certain boolean hyperparameters to a certain
value leads to other parameters being void of effect.

To choose whether to run the experiments on a CNN or MLP, the ARCHITECTURE global variable can be updated.

To create tensorboard graphs of the parameter gradients and other data, the writer_path variable can be set to the desired
location (e.g: "runs", "runs/mnist", etc.).
"""


import torch

from torch.utils.tensorboard import SummaryWriter

from data import get_train_dataloader_MNIST, get_test_dataloader_MNIST
from train import train
from test import test
from models import MNIST_MLP, MNIST_CNN
from activation import optimal_trelu_params

device = "cuda" if torch.cuda.is_available() else "cpu"


ARCHITECTURE = "MLP"  # "CNN"


if ARCHITECTURE not in ["MLP", "CNN"]:
    raise Exception("Network type can only be one of MLP or CNN")


# MODEL HYPERPARAMETERS
width = kernels = None
if ARCHITECTURE == "MLP":
    width = 100
else:
    kernels = 12
depth              = 100
activation         = "trelu"  # Can take values "relu", "trelu"
residual_connections_on = False
activation_before_residual = False  # No effect if residual branches are off
trelu_is_trainable = False  # If the activation is set to "trelu", determines whether the alpha parameter is trainable, otherwise has no effect
alpha              = 1.0  # If the activation is set to "trelu", this is the starting alpha parameter, otherwise has no effect
beta               = 0.5  # If residual branches are active, beta takes this value, otherwise has no effect
beta_is_trainable  = False
beta_is_global     = False  # If beta is not trainable has no effect, otherwise if this is True, beta takes the same value for every layer during training
normalize          = True  # Determines whether batch normalization is active

# TRAINING HYPERPARAMETERS
batch_size    = 256
epochs        = 5
adam_lr       = 1e-4 if ARCHITECTURE == "MLP" and depth >= 200 else 1e-3
adam_alpha_lr = 1e-2  # The ad-hoc learning rate to use for the alpha parameter of the ReLU (if trainable)
adam_beta_lr  = 1e-3  # The ad-hoc learning rate to use for the beta parameter of the residual branches (if trainable)

# BEST ALPHA COMPUTATION PARAMETERS
compute_best_alpha = True  # If set to True, computes the two optimal alpha values and overwrites the alpha set above with one of those values
eta                = 0.9


writer_path = ""
writer = SummaryWriter(writer_path) if writer_path != "" else None


if __name__ == "__main__":

    # If compute_best_alpha is True the value of alpha is overwritten with one of the optimal ones
    if compute_best_alpha:
        OPTIMAL_ALPHA_1, OPTIMAL_ALPHA_2 = list(optimal_trelu_params(depth, eta))
        alpha = OPTIMAL_ALPHA_1  # OPTIMAL_ALPHA_2

    # Create a model
    if ARCHITECTURE == "MLP":
        model = MNIST_MLP(depth,
                          width,
                          activation,
                          alpha,
                          device,
                          trelu_is_trainable=trelu_is_trainable,
                          residual_connections=residual_connections_on,
                          beta_init=beta,
                          beta_is_trainable=beta_is_trainable,
                          beta_is_global=beta_is_global,
                          activation_before_residual=activation_before_residual,
                          normalize=normalize).to(device)
    else:
        model = MNIST_CNN(depth,
                          kernels,
                          activation,
                          alpha,
                          device,
                          trelu_is_trainable=trelu_is_trainable,
                          residual_connections=residual_connections_on,
                          beta_init=beta,
                          beta_is_trainable=beta_is_trainable,
                          beta_is_global=beta_is_global,
                          activation_before_residual=activation_before_residual,
                          normalize=normalize).to(device)

    # Create loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = None
    if model.beta is not None:
        optimizer = torch.optim.Adam([
            {'params': model.base_params(), 'lr': adam_lr},
            {'params': model.beta, 'lr': adam_beta_lr},
            {'params': model.trelu_params(), 'lr': adam_alpha_lr}
        ])
    else:
        optimizer = torch.optim.Adam([
            {'params': model.base_params(), 'lr': adam_lr},
            {'params': model.trelu_params(), 'lr': adam_alpha_lr}
        ])

    # Train for n epochs
    train_dataloader = get_train_dataloader_MNIST(batch_size)
    test_dataloader = get_test_dataloader_MNIST(batch_size)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(train_dataloader, batch_size, model, criterion, optimizer, epoch, device, writer)
        test(test_dataloader, model, criterion, epoch, device, writer)
