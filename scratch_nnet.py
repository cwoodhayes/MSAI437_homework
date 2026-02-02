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
    def forward(self, *argc, **argv) -> np.ndarray | float:
        pass

    @abc.abstractmethod
    def backward(self, *argc, **argv) -> np.ndarray:
        pass

    @abc.abstractmethod
    def parameters(self) -> list[np.ndarray]:
        pass

    def __call__(self, X: np.ndarray) -> np.ndarray | float:
        return self.forward(X)

    # some functions that I'm declaring here just as dummy functions
    # to match the pytorch API
    def to(self, device: str) -> Module:
        """Dummy function to match pytorch API."""
        return self


class Sequential(Module):
    """Sequence of blocks."""

    def __init__(self, blocks: list[Module]) -> None:
        super().__init__()
        self._blocks = blocks

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for block in self._blocks:
            out = block(out)  # type: ignore
        return out  # type: ignore

    def backward(self, dout: np.ndarray) -> np.ndarray:
        grad = dout
        for block in reversed(self._blocks):
            grad = block.backward(grad)  # type: ignore
        return grad

    def parameters(self) -> list[np.ndarray]:
        params: list[np.ndarray] = []
        for block in self._blocks:
            params.extend(block.parameters())
        return params


class Tanh(Module):
    """TanH activation function block."""

    def __init__(self) -> None:
        super().__init__()
        self._last_output: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = np.tanh(X)
        self._last_output = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._last_output is None:
            raise RuntimeError("Tanh.backward() called before forward().")
        return dout * (1 - self._last_output**2)

    def parameters(self) -> list[np.ndarray]:
        return []


class Linear(Module):
    """single linear layer block."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self._last_input_h: np.ndarray | None = None
        self._grads: np.ndarray | None = None

        normal = np.random.default_rng().standard_normal

        # express params as a single matrix so we can use homogeneous
        # coordinates later and do it all in a single multiplication
        # this also helps with indexing the weight for backprop later
        self._params = normal((in_features + 1, out_features + 1)) * 0.01
        # last row is bias, so set last column except that to 0
        self._params[:-1, -1] = 0.0

    def forward(self, X: np.ndarray) -> np.ndarray:
        # each row of X is a data point
        n_samples = X.shape[0]
        # add 1 at the end of each data point for bias
        X_h = np.hstack((X, np.ones((n_samples, 1))))
        self._last_input_h = X_h
        out_h = X_h @ self._params
        return out_h[:, :-1]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._last_input_h is None:
            raise RuntimeError("Linear.backward() called before forward().")

        # Need to add homogeneous column to dout to match params shape
        n_samples = dout.shape[0]
        dout_h = np.hstack((dout, np.zeros((n_samples, 1))))

        # gradients for parameters
        self._grads = self._last_input_h.T @ dout_h

        # gradient w.r.t. input (drop homogeneous column)
        dX_h = dout_h @ self._params.T
        return dX_h[:, :-1]

    def parameters(self) -> list[np.ndarray]:
        return [self._params]


class Sigmoid(Module):
    """Sigmoid output layer."""

    def __init__(self) -> None:
        super().__init__()
        self._last_output: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = 1 / (1 + np.exp(-X))
        self._last_output = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._last_output is None:
            raise RuntimeError("Sigmoid.backward() called before forward().")
        return dout * self._last_output * (1 - self._last_output)

    def parameters(self) -> list[np.ndarray]:
        return []


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

    def backward(self, X: np.ndarray):
        return self._net.backward(X)

    def parameters(self) -> list[np.ndarray]:
        return self._net.parameters()
