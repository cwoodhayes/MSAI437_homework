"""
MSE-based neural network as in Q2, implemented from scratch.

doing this in a real .py file because I want good code organization,
since this is a bit more complicated.
"""

from __future__ import annotations
import abc
from dataclasses import dataclass

import numpy as np


class Module(abc.ABC):
    """Base class for my sub-network entities."""

    @abc.abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def backprop(self, X: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)


class Sequential(Module):
    """Sequence of blocks."""

    def __init__(self, blocks: list[Module]) -> None:
        super().__init__()
        self._blocks = blocks

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for block in self._blocks:
            out = block(out)
        return out

    def backprop(self, X: np.ndarray) -> np.ndarray:
        # TODO
        raise NotImplementedError


class Tanh(Module):
    """TanH activation function block."""

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X)

    def backprop(self, X: np.ndarray) -> np.ndarray:
        # TODO
        raise NotImplementedError


class Linear(Module):
    """single linear layer block."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        normal = np.random.default_rng().standard_normal
        self._weights = normal((in_features, out_features)) * 0.01
        self._bias = np.zeros((1, out_features))

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self._weights + self._bias

    def backprop(self, X: np.ndarray) -> np.ndarray:
        # TODO
        raise NotImplementedError


class Sigmoid(Module):
    """Sigmoid output layer."""

    def forward(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def backprop(self, X: np.ndarray) -> np.ndarray:
        # TODO
        raise NotImplementedError


class ScratchMSENet(Module):
    """
    1xk neural network implementation from scratch

    API attempts to mimic pytorch implementation as much as
    possible so I can reuse my other functions.
    """

    @dataclass(eq=True, order=True, unsafe_hash=True)
    class Hyperparams:
        """Hyperparameters for training + model"""

        k: int = 7
        """Number of nodes in the hidden layer. One of {2, 3, 5, 7, 9}"""

        learning_rate: float = 1e-3
        training_epochs: int = 5
        batch_size: int = 50

    def __init__(self, p: Hyperparams):
        self.hparams = p

        self._net = Sequential(
            [
                Linear(2, self.hparams.k),
                Tanh(),
                Linear(self.hparams.k, 1),
                Sigmoid(),
            ]
        )

    def forward(self, X: np.ndarray):
        return self._net(X)

    def backprop(self, X: np.ndarray):
        return self._net.backprop(X)
