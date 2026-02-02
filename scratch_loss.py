"""
Loss implementations.
"""

import numpy as np
from scratch_nnet import Module


class MSELoss(Module):
    """Mean squared error loss"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        return np.mean((y - yhat) ** 2)  # type: ignore

    def backward(
        self,
        y: np.ndarray,
        yhat: np.ndarray,
    ) -> np.ndarray:
        # Compute gradient of MSE loss with respect to predictions
        n = yhat.shape[0]
        return (2 / n) * (yhat - y)

    def parameters(self) -> list[np.ndarray]:
        return []

    def grads(self) -> list[np.ndarray]:
        return []
