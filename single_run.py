import torch

from torch.utils.tensorboard import SummaryWriter

from data import get_train_dataloader_MNIST, get_test_dataloader_MNIST
from train import train
from test import test
from model import MNIST_MLP
from activation import optimal_trelu_params

device = "cuda" if torch.cuda.is_available() else "cpu"


# MODEL HYPERPARAMETERS
depth              = 100
width              = 100
activation         = "trelu"
trelu_is_trainable = True
alpha              = 1.0
normalize          = True

# TRAINING HYPERPARAMETERS
batch_size    = 256
epochs        = 5
adam_lr       = 1e-3 if depth < 200 else 1e-4
adam_alpha_lr = 1e-2

# BEST ALPHA COMPUTATION PARAMETERS
eta                = 0.9
compute_best_alpha = False


if __name__ == "__main__":

    # If compute_best_alpha is True we use this value for alpha
    if compute_best_alpha:
        OPTIMAL_ALPHA_1, OPTIMAL_ALPHA_2 = list(optimal_trelu_params(depth, eta))
        alpha = OPTIMAL_ALPHA_1  # OPTIMAL_ALPHA_2

    writer_path = f"runs/depth{depth}/{activation}"
    if activation == "trelu":
        writer_path += "/trainable" if trelu_is_trainable else "/untrainable"
    writer = SummaryWriter(writer_path)
    writer = None

    # Create a model
    model = MNIST_MLP(device,
                      depth,
                      width,
                      activation,
                      alpha,
                      trelu_is_trainable=trelu_is_trainable,
                      normalize=normalize).to(device)

    # Create loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
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
