
from typing import Callable
import numpy as np

# Type aliases
activation_function_type = tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]]
loss_function_type = tuple[Callable[[np.ndarray, np.ndarray], float], Callable[[np.ndarray, np.ndarray], np.ndarray]]
learning_rate_optimizer_type = Callable[[float, int, np.ndarray, np.ndarray], float]

# Activation functions
ReLU: activation_function_type = (
    lambda x: np.maximum(0, x),
    lambda x, y: (x > 0) * y
)

Id: activation_function_type = (
    lambda x: x,
    lambda x, y: y
)

sigmoid: activation_function_type = (
    lambda x: 1 / (1 + np.exp(-x)),
    lambda x, y: (np.exp(-x) / (1 + np.exp(-x))**2) * y
)

tanh: activation_function_type = (
    lambda x: np.tanh(x),
    lambda x, y: (1 - np.tanh(x)**2) * y
)


def forward_softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exps    = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)

def backward_softmax(x: np.ndarray, delta: np.ndarray) -> np.ndarray:
    y = forward_softmax(x)
    return y * (delta - np.sum(delta * y, axis=1, keepdims=True))

softmax: activation_function_type = (
    forward_softmax, 
    backward_softmax
)

# Loss functions
mse: loss_function_type = (
    lambda x, y: np.mean(np.power(y - x, 2)),
    lambda x, y: 2 * (x - y) / x.shape[1]
)

# Learning rate optimizers
def fixed_learning_rate() -> learning_rate_optimizer_type:
    return lambda learning_rate, epoch, dW, db: learning_rate

def time_based_decay(func: Callable[[int], float]) -> learning_rate_optimizer_type:
    return lambda learning_rate, epoch, dW, db: func(epoch)

def exponential_decay(k: int | float) -> learning_rate_optimizer_type:
    return time_based_decay(lambda x: np.exp(-k * x))

def step_decay(decay_rate: float = 0.5, step: int = 10) -> learning_rate_optimizer_type:
    return lambda learning_rate, epoch, dW, db: learning_rate * decay_rate if epoch % step == 0 else learning_rate

