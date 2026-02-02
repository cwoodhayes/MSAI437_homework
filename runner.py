"""
Train & test runner for scratch-only nn implementation.

Mostly copied as-is from my original notebook implementation for pytorch nn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import matplotlib.colors
from matplotlib import pyplot as plt
import numpy as np

import scratch_loss as sloss
from scratch_nnet import ScratchMSENet
from scratch_optimizer import Optimizer


DatasetSplit = tuple[np.ndarray, np.ndarray]
DatasetDict = dict[str, DatasetSplit]


@dataclass
class RunResults:
    """Results from a training run."""

    model: ScratchMSENet
    data: DatasetDict

    test_accuracy: float = 0.0
    training_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)


def _normalize_labels(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.reshape(-1, 1)
    return y


def _iter_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


def train_loop(
    X: np.ndarray,
    y: np.ndarray,
    model: ScratchMSENet,
    optimizer: Optimizer,
    batch_size: int,
) -> float:
    """Train for one epoch. Returns last-batch loss (for parity with notebook)."""
    loss_fn = sloss.MSELoss()
    y = _normalize_labels(y)
    last_loss = 0.0
    for Xb, yb in _iter_batches(X, y, batch_size, shuffle=True):
        yhat = model.forward(Xb)
        last_loss = float(loss_fn.forward(yb, yhat))
        dloss = loss_fn.backward(yb, yhat)
        model.backward(dloss)
        grads = model.grads()
        optimizer.step(grads)
    return last_loss


def test_loop(
    X: np.ndarray,
    y: np.ndarray,
    model: ScratchMSENet,
    batch_size: int,
) -> tuple[float, float]:
    """Evaluate model. Returns (accuracy, avg_loss)."""
    loss_fn = sloss.MSELoss()
    y = _normalize_labels(y)
    n = X.shape[0]
    total_loss = 0.0
    correct = 0
    total = 0

    for Xb, yb in _iter_batches(X, y, batch_size, shuffle=False):
        yhat = model.forward(Xb)
        total_loss += float(loss_fn.forward(yb, yhat))
        preds = (yhat > 0.5).astype(int)
        correct += int((preds == yb).sum())
        total += yb.size

    avg_loss = total_loss / max(1, int(np.ceil(n / batch_size)))
    accuracy = correct / max(1, total)
    return accuracy, avg_loss


class TrainAndTestRun:
    """Scratch-only training run using provided model/optimizer and MSE loss."""

    def __init__(
        self,
        model: ScratchMSENet,
        optimizer: Optimizer,
        data: DatasetDict,
        dset_name: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.dset_name = dset_name
        self.results: RunResults | None = None

    def eval(self) -> RunResults:
        X_train, y_train = self.data["train"]
        X_test, y_test = self.data["test"]
        X_valid, y_valid = self.data["valid"]

        batch_size = int(self.model.hparams.batch_size)
        epochs = int(self.model.hparams.training_epochs)

        res = RunResults(self.model, self.data)

        test_acc = 0.0
        for _ in range(epochs):
            train_loss = train_loop(
                X_train, y_train, self.model, self.optimizer, batch_size
            )
            test_acc, _ = test_loop(X_test, y_test, self.model, batch_size)
            _, valid_loss = test_loop(X_valid, y_valid, self.model, batch_size)

            res.training_loss.append(train_loss)
            res.validation_loss.append(valid_loss)

        res.test_accuracy = test_acc
        self.results = res
        return res


########### Plotting helpers from example code
def compute_bounds(features):
    min1, max1 = features[:, 0].min() - 1, features[:, 0].max() + 1
    min2, max2 = features[:, 1].min() - 1, features[:, 1].max() + 1
    return (min1, max1, min2, max2)


def plot_decision_regions(
    features,
    targets,
    model,
    axis=None,
    transform=None,
    bounds=None,
    title="Decision Surface",
):
    """
    Slightly different plotting approach than above. Used in backprop demo.

    This function produces a single plot containing a scatter plot of the
    features, targets, and decision region of the model.

    Args:
        features (np.ndarray): 2D array containing real-valued inputs.
        targets (np.ndarray): 1D array containing binary targets.
        model: a learner with .predict() method
        axis: the axis on which to plot. If None, create a new plot
        title: title of the plot
    Returns:
        None (plots to the active figure)
    """

    # define bounds of the domain
    if bounds is None:
        min1, max1, min2, max2 = compute_bounds(features)
    else:
        min1, max1, min2, max2 = bounds

    # define grid for visualizing decision regions
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)

    xx, yy = np.meshgrid(x1grid, x2grid)

    # flatten grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    # horizontally stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1, r2))

    # if we're transforming the features, do that now
    #     this allows xx and yy to still be in 2D for the visualization
    #     but grid has been transformed so it matches up with the fit model
    if transform is not None:
        grid = transform(grid)

    # generate predictions over grid
    yhat = model(grid)

    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)

    if axis is None:
        fig, axis = plt.subplots()

    # plot the grid of x, y and z values as a surface
    binary_cmap = matplotlib.colors.ListedColormap(["#9ce8ff", "#ffc773"])
    axis.contourf(xx, yy, zz, cmap=binary_cmap, alpha=0.7)

    # plot "negative" class:
    row_idx_neg = np.where(targets < 0.5)[0]
    axis.scatter(features[row_idx_neg, 0], features[row_idx_neg, 1], label="negative")

    # plot "positive" class:
    row_idx_pos = np.where(targets > 0.5)[0]
    axis.scatter(features[row_idx_pos, 0], features[row_idx_pos, 1], label="positive")

    axis.set_title(title)
    axis.set_xlim(min1, max1)
    axis.set_ylim(min2, max2)

    axis.legend(loc="upper left")


################# end plotting helpers


def eval_and_plot(
    runs: Iterable[tuple[str, DatasetDict, ScratchMSENet, Optimizer]],
) -> None:
    """Evaluate scratch runs and plot learning curves + decision regions."""
    for dset_name, data, model, optimizer in runs:
        run = TrainAndTestRun(model, optimizer, data, dset_name)
        res = run.eval()

        fig = plt.figure()
        ax0, ax1 = fig.subplots(1, 2)

        epochs = int(res.model.hparams.training_epochs)
        ax0.plot(range(epochs), res.training_loss, label="Training Loss")
        ax0.plot(range(epochs), res.validation_loss, label="Validation Loss")
        ax0.set_xlabel("Epochs")
        ax0.set_ylabel("MSE loss")
        ax0.set_title("Learning Curve")
        ax0.legend()

        hparams_str = (
            str(res.model.hparams)
            .replace("ScratchMSENet.Hyperparams(", "")
            .replace(")", "")
        )
        fig.suptitle(
            f"(Scratch NN Implementation)\nDataset: {dset_name}\n{hparams_str}\nFinal Accuracy: {100 * res.test_accuracy:.1f}%"
        )

        X_test, y_test = res.data["test"]
        plot_decision_regions(
            X_test,
            y_test.ravel(),
            lambda x: (res.model.forward(x) > 0.5).astype(int).ravel(),
            axis=ax1,
            title="Decision Regions",
        )

        plt.tight_layout()
