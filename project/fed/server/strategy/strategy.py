"""FedAvg with Iterative Moving Average."""

from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common.typing import NDArrays
import numpy as np

from typing import Any


class IMA(FedAvg):
    """Federated Averaging with Iterative Moving Average.

    Add IMA to FedAvg
    """

    def __init__(
        self,
        window_size: int = 5,
        ima_start_round: int = 5,
        lr_decay: float = 0.01,
        ima_lr_decay: float = 0.03,
        *args: list[Any],
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.window_size = window_size
        if ima_start_round is None:
            ima_start_round = window_size
        self.ima_start_round = ima_start_round
        self.lr_decay = lr_decay
        self.ima_lr_decay = ima_lr_decay
        self.parameter_history: list[NDArrays] = []
        self.lr: float = (kwargs["on_fit_config_fn"])(0)["run_config"]["learning_rate"]  # type: ignore[operator]

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        if server_round == 1:
            self.lr = config["run_config"]["learning_rate"]  # type: ignore[index,call-overload,assignment]
        elif server_round < self.ima_start_round:
            self.lr *= 1 - self.lr_decay
        else:
            self.lr *= 1 - self.ima_lr_decay
        config["run_config"]["learning_rate"] = self.lr  # type: ignore[arg-type,index]

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        self.clients = clients
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:  # noqa: PLR2004
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # use the same clients as fitted for evaluation
        clients = self.clients

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """Aggregate fit results using weighted average and return IMA."""
        result = super().aggregate_fit(server_round, results, failures)

        if result[0] is None:
            return result

        params = result[0]

        if server_round >= self.ima_start_round - self.window_size:
            self.parameter_history += [parameters_to_ndarrays(params)]
            self.parameter_history = self.parameter_history[-self.window_size :]

        if server_round >= self.ima_start_round:
            n = len(self.parameter_history[0])
            moving_avg = [
                np.stack([weights[i] for weights in self.parameter_history])
                for i in range(n)
            ]
            moving_avg = [np.mean(weights, axis=0) for weights in moving_avg]
            params = ndarrays_to_parameters(moving_avg)

        return params, result[1]
