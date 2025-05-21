import numpy as np
import pytest

from encephalon.types_and_functions import (
    ReLU, Id, sigmoid, tanh, softmax, mse,
    forward_softmax, backward_softmax,
    fixed_learning_rate, exponential_decay, step_decay
)


@pytest.mark.parametrize("f,fprime", [ReLU, Id, sigmoid, tanh])
def test_activation_forward_and_backward_shapes(f, fprime):
    # single example
    x = np.random.randn(5, 3)
    y = f(x)
    assert y.shape == x.shape
    grad = fprime(x, np.ones_like(x))
    assert grad.shape == x.shape

def test_softmax_forward_rows_sum_to_1():
    X = np.random.randn(4, 6)
    Y = forward_softmax(X)
    # rows sum to 1
    np.testing.assert_allclose(Y.sum(axis=1), np.ones(4), atol=1e-6)

def test_softmax_backward_against_numerical():
    # small batch, small n for numeric diff
    X = np.random.randn(3, 4)
    delta = np.random.randn(3, 4)
    # analytic
    A = forward_softmax(X)
    grad = backward_softmax(X, delta)
    # numeric approximation for one example
    eps = 1e-5
    for i in range(3):
        for j in range(4):
            Xp = X.copy()
            Xp[i, j] += eps
            Ap = forward_softmax(Xp)
            loss_p = (Ap * delta).sum()
            Xm = X.copy()
            Xm[i, j] -= eps
            Am = forward_softmax(Xm)
            loss_m = (Am * delta).sum()
            numgrad = (loss_p - loss_m) / (2 * eps)
            assert abs(grad[i, j] - numgrad) < 1e-3

def test_mse_loss_and_grad():
    Y = np.array([[0.0, 1.0], [2.0, -1.0]])
    T = np.array([[1.0, 0.0], [1.0, 1.0]])
    loss = mse[0](Y, T)
    # manual: ((1+1)+(1+4)) / 4 = 7/4 = 1.75
    assert pytest.approx(1.75) == loss
    grad = mse[1](Y, T)  # shape (2,2)
    # derivative = 2*(Y-T)/n_features=2
    expected = 2 * (Y - T) / 2
    np.testing.assert_allclose(grad, expected)

def test_fixed_and_decay_schedules():
    base_lr = 0.1
    fl = fixed_learning_rate()
    assert fl(base_lr, 5, None, None) == base_lr
    ed = exponential_decay(0.5)
    assert ed(base_lr, 0, None, None) == pytest.approx(base_lr * np.exp(-0.5 * 0))
    assert ed(base_lr, 2, None, None) == pytest.approx(base_lr * np.exp(-0.5 * 2))
    sd = step_decay(decay_rate=0.1, step=3)
    assert sd(base_lr, 2, None, None) == base_lr
    assert sd(base_lr, 3, None, None) == pytest.approx(base_lr * 0.1)
