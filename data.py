from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


_training_data_MNIST = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

_test_data_MNIST = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


def get_train_dataloader_MNIST(batch_size):
    return DataLoader(_training_data_MNIST, batch_size=batch_size, shuffle=True)


def get_test_dataloader_MNIST(batch_size):
    return DataLoader(_test_data_MNIST, batch_size=batch_size)
