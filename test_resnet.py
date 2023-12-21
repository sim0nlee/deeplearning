import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import imgaug.augmenters as iaa

from resnet import ResNet

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"{device=}")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Augmentation using imgaug
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),         # Horizontal flip
    iaa.Crop(percent=(0, 0.1)),  # Random crop
    iaa.ScaleX((0.8, 1.2)),  # Scale
    iaa.Affine(rotate=(-20, 20)),  # Rotation
    iaa.ShearX((-16, 16)),    # Shear
    iaa.Multiply((0.7, 1.3)),  # Brightness
    iaa.contrast.LinearContrast((0.8, 1.2)),  # Contrast
    iaa.Dropout(0.2),         # Dropout
], random_order=True)


# Create DataLoader instances
train_dataset = ImageFolder('tiny-imagenet-200/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create DataLoader instance for test data
test_dataset = ImageFolder('tiny-imagenet-200/val', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# resnetX = (Num of channels, repetition, Bottleneck_expansion , Bottleneck_layer)
model_parameters = {}
model_parameters['resnet18'] = [[64, 128, 256, 512], [2, 2, 2, 2], 1, False]
model_parameters['resnet34'] = [[64, 128, 256, 512], [3, 4, 6, 3], 1, False]
model_parameters['resnet50'] = [[64, 128, 256, 512], [3, 4, 6, 3], 4, True]
model_parameters['resnet101'] = [[64, 128, 256, 512], [3, 4, 23, 3], 4, True]
model_parameters['resnet152'] = [[64, 128, 256, 512], [3, 8, 36, 3], 4, True]

# Create ResNet model
model = ResNet(model_parameters['resnet50'], in_channels=3, num_classes=200).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Change optimizer to Adam

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
num_epochs = 20

# Save a checkpoint every 2 epochs
checkpoint_interval = 2

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for step, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
        inputs, labels = inputs.to(torch.device("cpu")), labels.to(torch.device("cpu"))

       # Apply imgaug augmentation
        augmented_inputs = []
        augmented_labels = []
        for img, label in zip(inputs, labels):
            img = img.permute(1, 2, 0).numpy()  # Convert to HWC format
            img = augmentation.augment_image(img)
            img = torch.from_numpy(img.transpose(2, 0, 1).copy())  # Convert back to CHW format
            augmented_inputs.append(img)
            augmented_labels.append(label)
        augmented_inputs = torch.stack(augmented_inputs).to(device)
        augmented_labels = torch.stack(augmented_labels).to(device)

        optimizer.zero_grad()
        outputs = model(augmented_inputs)
        loss = criterion(outputs, augmented_labels)  # Use augmented labels in the loss calculation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == augmented_labels).sum().item()  # Use augmented labels for comparison
        total_samples += augmented_labels.size(0)

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
        checkpoint_path = os.path.join(checkpoint_dir, f'resnet_model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

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