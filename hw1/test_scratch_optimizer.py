"""Unit tests for scratch optimizer implementation."""

import numpy as np
import pytest
import scratch_nnet as snn
import scratch_loss as sloss
import scratch_optimizer as sopt


def test_sgd_single_step():
    """Test that SGD optimizer performs a single parameter update step."""
    # Create a simple model
    model = snn.ScratchMSENet(snn.ScratchMSENet.Hyperparams(k=3, learning_rate=0.01))

    # Store original parameters
    original_params = [p.copy() for p in model.parameters()]

    # Create optimizer
    optimizer = sopt.SGD(model.parameters(), learning_rate=0.1)

    # Forward pass
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y = np.array([[0.0], [1.0]], dtype=np.float32)
    yhat = model.forward(X)

    # Backward pass
    loss_fn = sloss.MSELoss()
    dloss = loss_fn.backward(y, yhat)
    model.backward(dloss)

    # Get gradients and perform optimization step
    grads = model.grads()
    optimizer.step(grads)

    # Check that parameters have changed
    updated_params = model.parameters()

    # At least one parameter should have changed
    params_changed = False
    for orig, updated in zip(original_params, updated_params):
        if not np.allclose(orig, updated):
            params_changed = True
            break

    assert params_changed, "Parameters should have been updated by optimizer"


def test_sgd_parameter_decrease_with_positive_gradients():
    """Test that SGD decreases parameters in direction of negative gradient."""
    # Create a simple Linear layer for easier testing
    layer = snn.Linear(2, 1)

    # Set known parameter values
    layer._params = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32)

    # Create optimizer
    optimizer = sopt.SGD([layer._params], learning_rate=0.1)

    # Create a dummy gradient (positive values)
    dummy_grad = np.ones_like(layer._params) * 0.1

    # Perform optimization step
    optimizer.step([dummy_grad])

    # Parameters should decrease (param -= lr * grad)
    expected = np.array([[0.49, 0.49, 0.49], [0.49, 0.49, 0.49]], dtype=np.float32)
    assert np.allclose(layer._params, expected, atol=1e-6), (
        f"Expected {expected}, got {layer._params}"
    )


def test_sgd_respects_learning_rate():
    """Test that SGD respects the learning rate parameter."""
    model = snn.ScratchMSENet(snn.ScratchMSENet.Hyperparams(k=3, learning_rate=0.01))

    original_params = [p.copy() for p in model.parameters()]

    # Create optimizer with high learning rate
    optimizer = sopt.SGD(model.parameters(), learning_rate=1.0)

    # Forward/backward pass
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y = np.array([[0.0], [1.0]], dtype=np.float32)
    yhat = model.forward(X)

    loss_fn = sloss.MSELoss()
    dloss = loss_fn.backward(y, yhat)
    model.backward(dloss)

    # Optimization step
    grads = model.grads()
    optimizer.step(grads)

    # Check magnitude of change (should be larger with higher LR)
    updated_params = model.parameters()
    max_change = 0.0
    for orig, updated in zip(original_params, updated_params):
        change = np.max(np.abs(updated - orig))
        max_change = max(max_change, change)

    # With LR=1.0, change should be noticeable (> 0.001)
    assert max_change > 0.001, (
        "Learning rate should produce noticeable parameter changes"
    )


def test_sgd_zero_grad_is_noop():
    """Test that zero_grad is a no-op for SGD."""
    model = snn.ScratchMSENet(snn.ScratchMSENet.Hyperparams(k=3, learning_rate=0.01))
    optimizer = sopt.SGD(model.parameters(), learning_rate=0.1)

    # zero_grad should return None and not raise an error
    result = optimizer.zero_grad()
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
