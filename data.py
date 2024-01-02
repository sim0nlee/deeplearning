from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import hyperparameters as hyps

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

train_dataloader = DataLoader(training_data, batch_size=hyps.batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=hyps.batch_size)