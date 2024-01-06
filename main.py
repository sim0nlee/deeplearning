import torch

from data import train_dataloader, test_dataloader
from train import train
from test import test
from model import MNIST_MLP, IMAGENET_CNN
from activation import optimal_trelu_params

import hyperparameters as hyps

from torch.utils.tensorboard import SummaryWriter

#run_name = f"runs/mnist/residual/adam/constant_beta/0.000001/50_epochs"
#run_name = f"runs/mnist/residual/adam/constant_beta/relu_pre_residual/0.001"
run_name = "runs/mnist/residual/adam/trainable_beta/lr_beta_1e-2/individual_betas/0.5/100_epochs/50_depth/shallow"
run_name = "runs/mnist/residual_cnn/adam/constant_beta/0.5/100_epochs/shallow_12"
run_name = "runs/mnist/residual_cnn/adam/trainable_beta_regularized_0.0001/lr_beta_1e-2/individual_betas/0.5/100_epochs/shallow_12"
#run_name = "runs/mnist/trelu_cnn/adam/100_epochs/12/lr_alpha_1e-3"
writer = SummaryWriter(run_name)

device = "cuda" if torch.cuda.is_available() else "cpu"

activation = "relu"  # "trelu"

OPTIMAL_ALPHA_1, OPTIMAL_ALPHA_2 = list(optimal_trelu_params())
alpha = 1.0
beta = 0.5

#model = MNIST_MLP(hyps.depth,
model = IMAGENET_CNN(hyps.depth,
                  hyps.width,
                  activation,
                  alpha,
                  device,
                  trelu_is_trainable=False,
                  residual_connections=True,
                  beta_init=beta,
                  beta_is_trainable=True,
                  beta_is_global=False,
                  activation_before_residual=False,
                  normalize=False).to(device)

#print(model)

criterion = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=hyps.sgd_lr)
#optimizer = torch.optim.Adam(model.parameters())
optimizer = torch.optim.Adam([
     {'params': model.base_params()},
     {'params': model.beta, 'lr': hyps.adam_beta_lr},
     #{'params': model.trelu_params(), 'lr': hyps.adam_alpha_lr}
])

if __name__ == "__main__":
    for epoch in range(hyps.epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, criterion, optimizer, epoch, device, writer)
        test(test_dataloader, model, criterion, epoch, device, writer)

    writer.close()