import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import hyperparameters as hyps
from resnet import ResNet

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"{device=}")

########################################################################################################################

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

########################################################################################################################

# Create DataLoader instances
train_dataset = ImageFolder('tiny-imagenet-200/train', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create DataLoader instance for test data
test_dataset = ImageFolder('tiny-imagenet-200/val', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

########################################################################################################################

# Create ResNet model
model = ResNet(resnet_variant='resnet18',
               in_channels=3,
               num_classes=200,
               activation="trelu",
               alpha_init=1.0,
               train_trelu=False,
               residual_connections=False,
               beta_init=0.5,
               beta_is_trainable=True,
               beta_is_global=False,
               normalize=False).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params': model.base_params()},
    *([{'params': model.beta, 'lr': hyps.adam_beta_lr}] if model.beta is not None else []),
    {'params': model.trelu_params(), 'lr': hyps.adam_alpha_lr}
])



# Set up TensorBoard for model graph logging
graph_writer = SummaryWriter('logs/tiny-imagenet-200/base/resnet50/graph')

# Add the model graph to TensorBoard
dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Create a dummy input with the same shape as your actual input
graph_writer.add_graph(model, dummy_input)

# Set up TensorBoard
writer = SummaryWriter('logs/tiny-imagenet-200/base/resnet50/train')

# Set up TensorBoard for test accuracy logging
test_writer = SummaryWriter('logs/tiny-imagenet-200/base/resnet50/test')

# Directory to save checkpoints
checkpoint_dir = 'checkpoints/base/resnet50'
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop
num_epochs = 15
checkpoint_interval = 1
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for step, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Log training loss and accuracy to TensorBoard
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + step)
        writer.add_scalar('Training Accuracy', correct_predictions / total_samples, epoch * len(train_loader) + step)

        # Log gradients of selected parameters to TensorBoard
        if step % 10 == 0:
            for name, param in model.named_parameters():
                if 'conv' in name and 'weight' in name:
                    writer.add_histogram(name + '_grad', param.grad, epoch * len(train_loader) + step)

    average_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}, Accuracy: {accuracy}")

    # Save checkpoint
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = os.path.join(checkpoint_dir, 'resnet_model.pth')  # Overwrite the same file
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

    # Validation (testing) loop
    model.eval()
    correct_predictions_test = 0
    total_samples_test = 0

    with torch.no_grad():
        for inputs_test, labels_test in tqdm(test_loader, desc=f'Testing Epoch {epoch + 1}/{num_epochs}'):
            inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)

            outputs_test = model(inputs_test)
            _, predicted_test = torch.max(outputs_test, 1)

            correct_predictions_test += (predicted_test == labels_test).sum().item()
            total_samples_test += labels_test.size(0)

    accuracy_test = correct_predictions_test / total_samples_test

    # Log test accuracy to TensorBoard
    test_writer.add_scalar('Test Accuracy', accuracy_test, epoch + 1)

    print(f"Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy_test}")

# Close TensorBoard writers
test_writer.close()
writer.close()
graph_writer.close()

# Save the final trained model
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
model_save_path = os.path.join(models_dir, 'resnet50_base.pth')
torch.save(model.state_dict(), model_save_path)
