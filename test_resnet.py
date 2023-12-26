import os

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Tuple

from resnet import ResNet

########################################################################################################################

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"{device=}")

########################################################################################################################

# resnetX = (Num of channels, repetition, Bottleneck_expansion , Bottleneck_layer)
model_parameters = {}
model_parameters['resnet18'] = [[64, 128, 256, 512], [2, 2, 2, 2], 1, False]
model_parameters['resnet34'] = [[64, 128, 256, 512], [3, 4, 6, 3], 1, False]
model_parameters['resnet50'] = [[64, 128, 256, 512], [3, 4, 6, 3], 4, True]
model_parameters['resnet101'] = [[64, 128, 256, 512], [3, 4, 23, 3], 4, True]
model_parameters['resnet152'] = [[64, 128, 256, 512], [3, 8, 36, 3], 4, True]

########################################################################################################################

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_dataset()

    def _load_dataset(self):
        samples = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue  # Skip if not a directory

            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                samples.append((image_path, self.class_to_idx[class_name]))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, target = self.samples[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, target

    def split_dataset(self, split_ratio: float = 0.8) -> Tuple[Dataset, Dataset, Dataset]:
        # Split the dataset into train, validation, and test sets
        train_size: int = int(split_ratio * len(self))
        val_test_size: int = len(self) - train_size
        train_dataset, val_test_dataset = random_split(self, [train_size, val_test_size])
        val_size = val_test_size // 2
        val_dataset, test_dataset = random_split(val_test_dataset, [val_size, val_size])
        return train_dataset, test_dataset, val_dataset

########################################################################################################################

# Define transformations for data augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor()])

########################################################################################################################

# Dataset loading
root_dir = 'imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC'
# root_dir = 'tiny-imagenet-200'
classes: int = 1000
# classes: int = 200
train_dataset, _, val_dataset = ImageNetDataset(root_dir, split='train', transform=transform).split_dataset()
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=30)  # batch_size=256, num_workers=30
test_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=30)  # batch_size=256, num_workers=30

########################################################################################################################

model = ResNet(model_parameters['resnet50'],
               in_channels=3,
               num_classes=classes,
               use_batch_norm=False,
               shortcut_weight=0.8,
               activation_name="relu").to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Set up TensorBoard
graph_writer = SummaryWriter('logs/imagenet/vanilla_with_shortcut/resnet50/graph')
writer = SummaryWriter('logs/imagenet/vanilla_with_shortcut/resnet50/train')
test_writer = SummaryWriter('logs/imagenet/vanilla_with_shortcut/resnet50/test')
grad_writer = SummaryWriter('logs/imagenet/vanilla_with_shortcut/resnet50/grad')
dummy_input = torch.randn(1, 3, 224, 224).to(device)
graph_writer.add_graph(model, dummy_input)

# Save checkpoints
checkpoint_dir = 'checkpoints/imagenet/vanilla_with_shortcut/resnet50'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_interval = 1

num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for step, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
        augmented_inputs, augmented_labels = inputs.to(device), labels.to(device)

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
                    # Log gradients norm
                    writer.add_histogram(name + '_grad', param.grad, epoch * len(train_loader) + step)
                    # Calculate and log gradients norm
                    grad_norm = torch.norm(param.grad)
                    grad_writer.add_scalar(name + '_grad_norm', grad_norm, epoch * len(train_loader) + step)

    # Update learning rate
    scheduler.step()

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

    # Testing loop
    model.eval()
    test_correct_predictions = 0
    test_total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f'Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_correct_predictions += (predicted == labels).sum().item()
            test_total_samples += labels.size(0)

    # Compute test accuracy
    test_accuracy = test_correct_predictions / test_total_samples

    # Log test accuracy to TensorBoard
    test_writer.add_scalar('Test Accuracy', test_accuracy, (epoch + 1) * len(train_loader))

# Close TensorBoard writers
writer.close()
graph_writer.close()
test_writer.close()
grad_writer.close()

# Save the final trained model
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
model_save_path = os.path.join(models_dir, 'resnet50_vanilla_with_shortcut.pth')
torch.save(model.state_dict(), model_save_path)
