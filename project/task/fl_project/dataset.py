"""CIFAR-10 dataset utilities for federated learning."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from project.task.default.dataset import (
    ClientDataloaderConfig as DefaultClientDataloaderConfig,
)
from project.task.default.dataset import (
    FedDataloaderConfig as DefaultFedDataloaderConfig,
)
from project.types.common import (
    CID,
    ClientDataloaderGen,
    FedDataloaderGen,
    IsolatedRNG,
)

# Use defaults for this very simple dataset
# Requires only batch size
ClientDataloaderConfig = DefaultClientDataloaderConfig
FedDataloaderConfig = DefaultFedDataloaderConfig


def get_dataloader_generators(
    partition_dir: Path,
    federated_dir: Path | None = None,
) -> tuple[ClientDataloaderGen, FedDataloaderGen]:
    """Return a function that loads a client's dataset.

    Parameters
    ----------
    partition_dir : Path
        The path to the partition directory.
        Containing the training data of clients.
        Partitioned by client id.

    Returns
    -------
    Tuple[ClientDataloaderGen, FedDataloaderGen]
        A tuple of functions that return a DataLoader for a client's dataset
        and a DataLoader for the federated dataset.
    """
    # for the case where the directory structure is not a simple
    # parition_dir
    #  |-- client_n
    #  |    |-- train.pt
    #  |    |-- val.pt
    #  |-- train.pt
    #  |-- test.pt
    # in our case the client folders are nested under the additional `concentration`
    if federated_dir is None:
        federated_dir = partition_dir

    def get_client_dataloader(
        cid: CID, test: bool, _config: dict, rng_tuple: IsolatedRNG
    ) -> DataLoader:
        """Return a DataLoader for a client's dataset.

        Parameters
        ----------
        cid : str|int
            The client's ID
        test : bool
            Whether to load the test set or not
        _config : Dict
            The configuration for the dataset
        rng_tuple : IsolatedRNGTuple
            The random number generator state for the training.
            Use if you need seeded random behavior

        Returns
        -------
        DataLoader
            The DataLoader for the client's dataset
        """
        config: ClientDataloaderConfig = ClientDataloaderConfig(**_config)
        del _config

        torch_cpu_generator = rng_tuple[3]

        client_dir = partition_dir / f"client_{cid}"
        if not test:
            dataset = torch.load(client_dir / "train.pt")
        else:
            dataset = torch.load(client_dir / "val.pt")
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=not test,
            generator=torch_cpu_generator,
        )

    def get_federated_dataloader(
        test: bool, _config: dict, rng_tuple: IsolatedRNG
    ) -> DataLoader:
        """Return a DataLoader for federated train/test sets.

        Parameters
        ----------
        test : bool
            Whether to load the test set or not
        config : Dict
            The configuration for the dataset
        rng_tuple : IsolatedRNGTuple
            The random number generator state for the training.
            Use if you need seeded random behavior

        Returns
        -------
            DataLoader
            The DataLoader for the federated dataset
        """
        config: FedDataloaderConfig = FedDataloaderConfig(
            **_config,
        )
        del _config
        torch_cpu_generator = rng_tuple[3]

        if not test:
            return DataLoader(
                torch.load(federated_dir / "train.pt"),
                batch_size=config.batch_size,
                shuffle=not test,
                generator=torch_cpu_generator,
            )

        return DataLoader(
            torch.load(federated_dir / "test.pt"),
            batch_size=config.batch_size,
            shuffle=not test,
            generator=torch_cpu_generator,
        )

    return get_client_dataloader, get_federated_dataloader


def get_dataloader_generators_centralised_test(
    partition_dir: Path,
    federated_dir: Path | None = None,
) -> tuple[ClientDataloaderGen, FedDataloaderGen]:
    """Return a function that loads a client's dataset.

    Parameters
    ----------
    partition_dir : Path
        The path to the partition directory.
        Containing the training data of clients.
        Partitioned by client id.

    Returns
    -------
    Tuple[ClientDataloaderGen, FedDataloaderGen]
        A tuple of functions that return a DataLoader for a client's dataset
        and a DataLoader for the federated dataset.
    """
    # for the case where the directory structure is not a simple
    # parition_dir
    #  |-- client_n
    #  |    |-- train.pt
    #  |    |-- val.pt
    #  |-- train.pt
    #  |-- test.pt
    # in our case the client folders are nested under the additional `concentration`
    if federated_dir is None:
        federated_dir = partition_dir

    def get_client_dataloader(
        cid: CID, test: bool, _config: dict, rng_tuple: IsolatedRNG
    ) -> DataLoader:
        """Return a DataLoader for a client's dataset.

        Parameters
        ----------
        cid : str|int
            The client's ID
        test : bool
            Whether to load the test set or not
        _config : Dict
            The configuration for the dataset
        rng_tuple : IsolatedRNGTuple
            The random number generator state for the training.
            Use if you need seeded random behavior

        Returns
        -------
        DataLoader
            The DataLoader for the client's dataset
        """
        config: ClientDataloaderConfig = ClientDataloaderConfig(**_config)
        del _config

        torch_cpu_generator = rng_tuple[3]

        client_dir = partition_dir / f"client_{cid}"
        if not test:
            dataset = torch.load(client_dir / "train.pt")
        else:
            dataset = torch.load(federated_dir / "test.pt")

        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=not test,
            generator=torch_cpu_generator,
        )

    def get_federated_dataloader(
        test: bool, _config: dict, rng_tuple: IsolatedRNG
    ) -> DataLoader:
        """Return a DataLoader for federated train/test sets.

        Parameters
        ----------
        test : bool
            Whether to load the test set or not
        config : Dict
            The configuration for the dataset
        rng_tuple : IsolatedRNGTuple
            The random number generator state for the training.
            Use if you need seeded random behavior

        Returns
        -------
            DataLoader
            The DataLoader for the federated dataset
        """
        config: FedDataloaderConfig = FedDataloaderConfig(
            **_config,
        )
        del _config
        torch_cpu_generator = rng_tuple[3]

        if not test:
            return DataLoader(
                torch.load(federated_dir / "train.pt"),
                batch_size=config.batch_size,
                shuffle=not test,
                generator=torch_cpu_generator,
            )

        return DataLoader(
            torch.load(federated_dir / "test.pt"),
            batch_size=config.batch_size,
            shuffle=not test,
            generator=torch_cpu_generator,
        )

    return get_client_dataloader, get_federated_dataloader
