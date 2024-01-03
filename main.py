import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from resnet import ResNet
from train import train
from test import test

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"{device=}")

# Define transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

# Run data_organize.py in tiny-imagenet-200

# Create DataLoader instances
train_dataset = ImageFolder('tiny-imagenet-200/train', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=12)
test_dataset = ImageFolder('tiny-imagenet-200/val/images', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=12)

print(train_dataset.classes)
print(test_dataset.classes)

# Create ResNet model
model = ResNet(resnet_variant='resnet18',
               in_channels=3,
               num_classes=200,
               activation="relu",
               alpha_init=1.0,
               train_trelu=False,
               scale_residual_connections=False,
               beta_init=0.5,
               beta_is_trainable=False,
               beta_is_global=False,
               normalize=True).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params': model.base_params()},
    *([{'params': model.beta}] if model.beta is not None else []),
    {'params': model.trelu_params()}
], lr=1e-2)

# Decay LR by a factor of 0.1 every 5 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# Set the directories for logs, checkpoints and final models
graph_dir = 'logs/tiny-imagenet-200/resnet18/relu__standard_connection__with_norm/graph'
train_dir = 'logs/tiny-imagenet-200/resnet18/relu__standard_connection__with_norm/train'
test_dir = 'logs/tiny-imagenet-200/resnet18/relu__standard_connection__with_norm/test'
checkpoint_dir = 'checkpoints/tiny-imagenet-200/resnet18/relu__standard_connection__with_norm'
final_model_dir = 'models/tiny-imagenet-200/resnet18/relu__standard_connection__with_norm'

# Check and create directories if they don't exist
os.makedirs(graph_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(final_model_dir, exist_ok=True)

# Set up TensorBoard for model graph logging
graph_writer = SummaryWriter(graph_dir)
dummy_input = torch.randn(1, 3, 64, 64).to(device)
graph_writer.add_graph(model, dummy_input)
train_writer = SummaryWriter(train_dir)
test_writer = SummaryWriter(test_dir)
os.makedirs(checkpoint_dir, exist_ok=True)

num_epochs = 15
checkpoint_interval = 1
if __name__ == "__main__":
    for epoch in range(num_epochs):
        train(train_loader, model, num_epochs, epoch, device, optimizer, criterion, train_writer, checkpoint_interval, checkpoint_dir)
        test(test_loader, model, num_epochs, epoch, device, test_writer)

    # Close TensorBoard writers
    test_writer.close()
    train_writer.close()
    graph_writer.close()

    # Save the final trained model
    model_save_path = os.path.join(final_model_dir, 'final_model.pth')
    torch.save(model.state_dict(), model_save_path)