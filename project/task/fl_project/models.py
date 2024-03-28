"""CNN model architecture, training, and testing functions for CIFAR-10."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import resnet18

from project.types.common import NetGen
from project.utils.utils import lazy_config_wrapper


class Net(nn.Module):
    """2 layer CNN.

    Based on IMA Paper Understanding and Improving Model Averaging
    in Federated Learning on Heterogeneous Data
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        """Run network."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    """3 layer CNN.

    Based on https://www.tensorflow.org/tutorials/images/cnn
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: Tensor) -> Tensor:
        """Run network."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Simple wrapper to match the NetGenerator Interface
get_net: NetGen = lazy_config_wrapper(Net)
get_cnn: NetGen = lazy_config_wrapper(CNN)

get_resnet: NetGen = lambda _config, _rng_tuple: resnet18(  # noqa: E731
    weights=None, progress=False, num_classes=10
)
