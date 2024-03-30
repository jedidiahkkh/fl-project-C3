"""Functions for CIFAR10 download and processing."""

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from flwr.common.logger import log
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

from project.fed.utils.lda_utils import create_lda_partitions


def _download_data(
    dataset_dir: Path,
) -> tuple[CIFAR10, CIFAR10]:
    """Download (if necessary) and returns the CIFAR10 dataset.

    Returns
    -------
    Tuple[CIFAR10, CIFAR10]
        The dataset for training and the dataset for testing CIFAR10.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    dataset_dir.mkdir(parents=True, exist_ok=True)

    trainset = CIFAR10(
        str(dataset_dir),
        train=True,
        download=True,
        transform=transform,
    )
    testset = CIFAR10(
        str(dataset_dir),
        train=False,
        download=True,
        transform=transform,
    )
    return trainset, testset


# pylint: disable=too-many-locals
def _partition_data(
    trainset: CIFAR10,
    testset: CIFAR10,
    num_clients: int,
    seed: int,
    concentration: float,
) -> tuple[list[Subset] | list[ConcatDataset], CIFAR10]:
    """Split training set into iid or non iid partitions to simulate the federated.

    setting.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    concentration: float
        Concentraion value for LDA
    seed : int
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[CIFAR10], CIFAR10]
        A list of dataset for each client and a single dataset to be used for testing
        the model.
    """
    x = np.array([x[0] for x in trainset])
    y = np.array(trainset.targets)

    partitions, _dist = create_lda_partitions(
        dataset=(x, y),
        dirichlet_dist=None,
        num_partitions=num_clients,
        concentration=concentration,
        accept_imbalanced=True,
        seed=seed,
    )

    return [list(zip(x, y, strict=True)) for x, y in partitions], testset


@hydra.main(
    config_path="../../conf",
    config_name="cifar",
    version_base=None,
)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Download and preprocess the dataset.

    Please include here all the logic
    Please use the Hydra config style as much as possible specially
    for parts that can be customized (e.g. how data is partitioned)

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. print parsed config
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    # Download the dataset
    trainset, testset = _download_data(
        Path(cfg.dataset.dataset_dir),
    )

    # Save the centralized test set
    # a centralized training set would also be possible
    # but is not used here
    partition_dir = Path(cfg.dataset.partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)
    centralized_client = partition_dir / "client_0"
    centralized_client.mkdir(parents=True, exist_ok=True)
    len_val = int(
        len(trainset) / (1 / cfg.dataset.val_ratio),
    )
    lengths = [len(trainset) - len_val, len_val]
    ds_train, ds_val = random_split(
        trainset,
        lengths,
        torch.Generator().manual_seed(cfg.dataset.seed),
    )
    torch.save(ds_train, centralized_client / "train.pt")
    torch.save(ds_val, centralized_client / "val.pt")
    torch.save(testset, partition_dir / "test.pt")

    # Partition the dataset
    # ideally, the fed_test_set can be composed in three ways:
    # 1. fed_test_set = centralized test set like CIFAR10
    # 2. fed_test_set = concatenation of all test sets of all clients
    # 3. fed_test_set = test sets of reserved unseen clients
    for c in cfg.dataset.concentrations:
        client_datasets, _ = _partition_data(
            trainset=trainset,
            testset=testset,
            num_clients=cfg.dataset.num_clients,
            seed=cfg.dataset.seed,
            concentration=c,
        )

        # 2. Save the datasets for this concentration value
        # unnecessary for this small dataset, but useful for large datasets
        partition_dir = Path(cfg.dataset.partition_dir) / f"{c}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        # Save the client datasets
        for idx, client_dataset in enumerate(client_datasets):
            client_dir = partition_dir / f"client_{idx}"
            client_dir.mkdir(parents=True, exist_ok=True)

            # # Alternative would have been to create train/test split
            # # when the dataloader is instantiated
            torch.save(client_dataset, client_dir / "train.pt")
            # torch.save(ds_val, client_dir / "test.pt")


if __name__ == "__main__":
    download_and_preprocess()
