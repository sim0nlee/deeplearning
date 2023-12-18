import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from resnet import ResNet

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create DataLoader instances
train_dataset = ImageFolder('tiny-imagenet-200/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

########################################################################################################################

# resnetX = (Num of channels, repetition, Bottleneck_expansion , Bottleneck_layer)
model_parameters={}
model_parameters['resnet18'] = [[64, 128, 256, 512], [2, 2, 2, 2], 1, False]
model_parameters['resnet34'] = [[64, 128, 256, 512], [3, 4, 6, 3], 1, False]
model_parameters['resnet50'] = [[64, 128, 256, 512], [3, 4, 6, 3], 4, True]
model_parameters['resnet101'] = [[64, 128, 256, 512], [3, 4, 23, 3], 4, True]
model_parameters['resnet152'] = [[64, 128, 256, 512], [3, 8, 36, 3], 4, True]

########################################################################################################################

# Create ResNet model
model = ResNet(model_parameters['resnet18'], in_channels=3, num_classes=200).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Set up TensorBoard
writer = SummaryWriter('logs/tiny-imagenet-200/base')

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for step, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Log training loss to TensorBoard
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + step)

        # Log gradients of selected parameters to TensorBoard
        if step % 10 == 0:
            for name, param in model.named_parameters():
                if 'conv' in name and 'weight' in name:  # Choose the layers you want to log
                    writer.add_histogram(name + '_grad', param.grad, epoch * len(train_loader) + step)

    average_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")

# Close TensorBoard writer
writer.close()

# Save the trained model if needed
torch.save(model.state_dict(), 'resnet_model.pth')
