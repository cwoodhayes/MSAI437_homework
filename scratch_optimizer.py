"""Simple optimizer implementations for scratch networks."""

from __future__ import annotations

import abc
from typing import List

import numpy as np


class Optimizer(abc.ABC):
    """Base class for optimizers."""

    @abc.abstractmethod
    def step(self, grads: List[np.ndarray]) -> None:
        """Apply one optimization step using gradients."""
        raise NotImplementedError

    @abc.abstractmethod
    def zero_grad(self) -> None:
        """Clear gradients (no-op for scratch implementation)."""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent optimizer."""

    def __init__(self, params: List[np.ndarray], learning_rate: float) -> None:
        self._params = params
        self._lr = learning_rate

    def step(self, grads: List[np.ndarray]) -> None:
        if len(grads) != len(self._params):
            raise ValueError(
                "SGD.step() expected grads list to match params list length."
            )
        for param, grad in zip(self._params, grads, strict=True):
            param -= self._lr * grad

    def zero_grad(self) -> None:
        return None
