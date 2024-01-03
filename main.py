import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter

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
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Create DataLoader instances
train_dataset = ImageFolder('tiny-imagenet-200/train', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset = ImageFolder('tiny-imagenet-200/val', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Create ResNet model
model = ResNet(resnet_variant='resnet18',
               in_channels=3,
               num_classes=200,
               activation="trelu",
               alpha_init=1.0,
               train_trelu=True,
               residual_connections=True,
               beta_init=0.5,
               beta_is_trainable=True,
               beta_is_global=False,
               normalize=False).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params': model.base_params()},
    *([{'params': model.beta, 'lr': 1e-2}] if model.beta is not None else []),
    {'params': model.trelu_params(), 'lr': 1e-2}
])

# Set the directories for logs, checkpoints and final models
graph_dir = 'logs/tiny-imagenet-200/base/resnet18/graph'
train_dir = 'logs/tiny-imagenet-200/base/resnet18/train'
test_dir = 'logs/tiny-imagenet-200/base/resnet18/test'
checkpoint_dir = 'checkpoints/tiny-imagenet-200/base/resnet18'
final_model_dir = 'models/tiny-imagenet-200/base/resnet18'

# Set up TensorBoard for model graph logging
graph_writer = SummaryWriter(graph_dir)
dummy_input = torch.randn(1, 3, 224, 224).to(device)
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
    os.makedirs(final_model_dir, exist_ok=True)
    model_save_path = os.path.join(final_model_dir, 'final_model.pth')
    torch.save(model.state_dict(), model_save_path)