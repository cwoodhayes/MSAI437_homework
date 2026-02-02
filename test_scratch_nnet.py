"""
Tests for scratch neural network implementation.
"""

import numpy as np
import pytest
from scratch_nnet import Linear, Tanh, Sigmoid, Sequential, ScratchMSENet
from scratch_loss import MSELoss


def test_linear_forward_shape():
    """Test Linear layer forward pass shape."""
    layer = Linear(2, 3)
    X = np.random.randn(5, 2)
    out = layer.forward(X)
    assert out.shape == (5, 3)


def test_linear_backward_shape():
    """Test Linear layer backward pass shapes."""
    layer = Linear(2, 3)
    X = np.random.randn(5, 2)
    out = layer.forward(X)
    dout = np.random.randn(5, 3)
    dX = layer.backward(dout)
    assert dX.shape == (5, 2)
    assert layer._grads is not None
    assert layer._grads.shape == (3, 4)  # (in+1, out+1) for homogeneous


def test_tanh_forward():
    """Test Tanh activation forward pass."""
    layer = Tanh()
    X = np.array([[0.0, 1.0], [-1.0, 0.5]])
    out = layer.forward(X)
    expected = np.tanh(X)
    np.testing.assert_allclose(out, expected)


def test_tanh_backward():
    """Test Tanh activation backward pass."""
    layer = Tanh()
    X = np.array([[0.5, -0.5], [1.0, -1.0]])
    out = layer.forward(X)
    dout = np.ones_like(out)
    dX = layer.backward(dout)
    # derivative of tanh is 1 - tanh^2
    expected = 1 - out**2
    np.testing.assert_allclose(dX, expected)


def test_sigmoid_forward():
    """Test Sigmoid activation forward pass."""
    layer = Sigmoid()
    X = np.array([[0.0, 1.0], [-1.0, 0.5]])
    out = layer.forward(X)
    expected = 1 / (1 + np.exp(-X))
    np.testing.assert_allclose(out, expected)


def test_sigmoid_backward():
    """Test Sigmoid activation backward pass."""
    layer = Sigmoid()
    X = np.array([[0.5, -0.5], [1.0, -1.0]])
    out = layer.forward(X)
    dout = np.ones_like(out)
    dX = layer.backward(dout)
    # derivative of sigmoid is sigmoid * (1 - sigmoid)
    expected = out * (1 - out)
    np.testing.assert_allclose(dX, expected)


def test_sequential_forward():
    """Test Sequential forward pass."""
    net = Sequential(
        [
            Linear(2, 3),
            Tanh(),
            Linear(3, 1),
        ]
    )
    X = np.random.randn(5, 2)
    out = net.forward(X)
    assert out.shape == (5, 1)


def test_sequential_backward():
    """Test Sequential backward pass."""
    net = Sequential(
        [
            Linear(2, 3),
            Tanh(),
            Linear(3, 1),
        ]
    )
    X = np.random.randn(5, 2)
    out = net.forward(X)
    dout = np.random.randn(5, 1)
    dX = net.backward(dout)
    assert dX.shape == (5, 2)


def test_backward_before_forward_raises():
    """Test that calling backward before forward raises RuntimeError."""
    layer = Tanh()
    dout = np.ones((5, 2))
    with pytest.raises(RuntimeError):
        layer.backward(dout)


def test_mse_loss_forward():
    """Test MSE loss forward pass."""
    loss_fn = MSELoss()
    y = np.array([[1.0], [0.0], [1.0]])
    yhat = np.array([[0.9], [0.1], [0.8]])
    loss = loss_fn.forward(y, yhat)
    expected = np.mean((y - yhat) ** 2)
    assert np.isclose(loss, expected)


def test_mse_loss_backward():
    """Test MSE loss backward pass."""
    loss_fn = MSELoss()
    y = np.array([[1.0], [0.0], [1.0]])
    yhat = np.array([[0.9], [0.1], [0.8]])
    loss_fn.forward(y, yhat)
    grad = loss_fn.backward(y, yhat)
    assert grad.shape == yhat.shape
    # gradient should be (2/n) * (yhat - y)
    expected = (2 / len(y)) * (yhat - y)
    np.testing.assert_allclose(grad, expected)


def test_scratch_msenet_forward():
    """Test full ScratchMSENet forward pass."""
    net = ScratchMSENet(ScratchMSENet.Hyperparams(k=5))
    X = np.random.randn(10, 2)
    out = net.forward(X)
    assert out.shape == (10, 1)
    # output should be between 0 and 1 (sigmoid)
    assert np.all((out >= 0) & (out <= 1))


def test_scratch_msenet_parameters():
    """Test that parameters are correctly returned."""
    net = ScratchMSENet(ScratchMSENet.Hyperparams(k=5))
    params = net.parameters()
    # should have 2 Linear layers, each with one param matrix
    assert len(params) == 2
    # first layer: (2+1, 5+1)
    assert params[0].shape == (3, 6)
    # second layer: (5+1, 1+1)
    assert params[1].shape == (6, 2)


def test_numerical_gradient_linear():
    """Test Linear backward against numerical gradient."""
    np.random.seed(42)
    layer = Linear(2, 3)
    X = np.random.randn(4, 2)

    # Forward and backward
    out = layer.forward(X)
    dout = np.random.randn(4, 3)
    dX_analytic = layer.backward(dout)

    # Numerical gradient for input
    eps = 1e-5
    dX_numerical = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_plus = X.copy()
            X_plus[i, j] += eps
            X_minus = X.copy()
            X_minus[i, j] -= eps

            # Need to re-forward with perturbed inputs
            layer._last_input_h = np.hstack((X_plus, np.ones((X_plus.shape[0], 1))))
            out_plus = X_plus @ layer._params[:2, :3] + layer._params[2:3, :3]
            layer._last_input_h = np.hstack((X_minus, np.ones((X_minus.shape[0], 1))))
            out_minus = X_minus @ layer._params[:2, :3] + layer._params[2:3, :3]

            dX_numerical[i, j] = np.sum(dout * (out_plus - out_minus)) / (2 * eps)

    np.testing.assert_allclose(dX_analytic, dX_numerical, rtol=1e-4, atol=1e-6)


def test_end_to_end_gradient_flow():
    """Test that gradients flow through entire network."""
    np.random.seed(42)
    net = ScratchMSENet(ScratchMSENet.Hyperparams(k=3))
    loss_fn = MSELoss()

    X = np.random.randn(5, 2)
    y = np.random.randint(0, 2, (5, 1)).astype(float)

    # Forward pass
    yhat = net.forward(X)
    loss = loss_fn.forward(y, yhat)

    # Backward pass
    dloss = loss_fn.backward(y, yhat)
    dX = net.backward(dloss)

    # Check shapes
    assert dX.shape == X.shape
    assert loss > 0

    # Check that gradients were computed for parameters
    params = net.parameters()
    linear_layers = [net._net._blocks[0], net._net._blocks[2]]
    for layer in linear_layers:
        assert layer._grads is not None
        assert layer._grads.shape == layer._params.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
