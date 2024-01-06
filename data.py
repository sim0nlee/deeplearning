from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import hyperparameters as hyps

training_data = datasets.MNIST(
#training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
#test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

train_dataloader = DataLoader(training_data, batch_size=hyps.batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=hyps.batch_size)

# # Create DataLoader instances
# train_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomAffine(degrees=(-10, 10), shear=(-5, 5)),
#     transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
#     transforms.RandomApply([transforms.RandomGrayscale(p=0.1)], p=0.1),
#     transforms.ToTensor()
# ])
# test_transform = transforms.Compose([
#      transforms.ToTensor()
# ])

# train_dataset = ImageFolder('data/tiny-imagenet-200/train', transform=train_transform)
# train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
# test_dataset = ImageFolder('data/tiny-imagenet-200/val/images', transform=test_transform)
# test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)