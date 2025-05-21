import json
import os
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from serial_utils import SerialJSONInterface, SerialProtocolError, SerialTimeoutError
from types_and_functions import (
    activation_function_type,
    loss_function_type,
    learning_rate_optimizer_type,
    ReLU,
    Id,
    mse,
    fixed_learning_rate,
)


class NN:
    def __init__(
        self,
        layers: list[int],
        serial_interface: SerialJSONInterface,
        name: str = "unnamed_model",
        f: activation_function_type = ReLU,
        g: activation_function_type = Id,
        verbose: bool = False,
    ) -> None:
        
        self.length = len(layers)
        if self.length < 2:
            raise ValueError("Network must have at least 2 layers (input and output).")

        self.serial = serial_interface
        self.name = name
        self.verbose = verbose

        # Activation functions
        self.f, self.backward_f = f
        self.g, self.backward_g = g

        # Weights and biases: layer i connects layers[i-1] to layers[i]
        self.W = {
            i: np.random.randn(layers[i - 1], layers[i]) * np.sqrt(2 / layers[i - 1])
            for i in range(1, self.length)
        }
        self.b = {i: np.zeros((1, layers[i])) for i in range(1, self.length)}

        self._clip_weights_and_biases()
        self.set_weights_and_biases()
        self._log("Weights and biases set.")
    

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.name}] {msg}")

    def _clip_weights_and_biases(self, min_value: float = -2.5, max_value: float = 2.5) -> None:
        for i in self.W:
            self.W[i] = np.clip(self.W[i], min_value, max_value)
            self.b[i] = np.clip(self.b[i], min_value, max_value)

    def set_weights_and_biases(self) -> None:
        payload = {
            "cmd": "set_weights_and_biases",
            "W": {str(k): v.tolist() for k, v in self.W.items()},
            "b": {str(k): v.tolist() for k, v in self.b.items()},
        }
        response = self.serial.send_and_receive(payload, expected_keys=["status"])
        if response["status"] != "OK":
            raise SerialProtocolError(f"Arduino failed to load weights and biases: {response}")

    def use(self, X: np.ndarray) -> np.ndarray:
        response = self.serial.send_and_receive({
            "cmd":"forward",
            "input": np.array(X, ndmin=2).tolist()         
        }, expected_keys=["output"])
        return self.g(np.array(response["output"]))

    def _forward_propagation(self, X: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        cache = []
        A = np.array(X, ndmin=2)
        for i in self.W:
            Z = A @ self.W[i] + self.b[i]
            cache.append((A, Z))
            A = self.f(Z)
        Z = A @ self.W[len(self.layers) - 1] + self.b[len(self.layers) - 1]
        cache.append((A, Z))
        return self.use(X), cache

    def _empty_grad(self) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        return (
            {i: np.zeros_like(self.W[i]) for i in self.W},
            {i: np.zeros_like(self.b[i]) for i in self.b},
        )

    def _add_grad(
        self,
        dW: dict[int, np.ndarray],
        db: dict[int, np.ndarray],
        grads: tuple[dict[int, np.ndarray], dict[int, np.ndarray]],
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        for i in dW:
            dW[i] += grads[0][i]
            db[i] += grads[1][i]
        return dW, db

    def _divide_grad(
        self, 
        dW: dict[int, np.ndarray], 
        db: dict[int, np.ndarray], 
        n: Union[int, float]
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        for i in dW:
            dW[i] /= n
            db[i] /= n
        return dW, db

import numpy as np
from typing import Callable

def _backward_propagation(
    self,
    output: np.ndarray,    # shape (m, n_L)
    label: np.ndarray,     # shape (m, n_L)
    cache: list[tuple[np.ndarray, np.ndarray]],  # [(A⁽⁰⁾, Z⁽¹⁾), …, (A⁽L⁻¹⁾, Z⁽L⁾)]
    backward_loss: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:

    m = output.shape[0]             # batch size
    dW, db = self._empty_grad()     # each dW[i], db[i] zero-arrays matching W[i], b[i]
    L = self.length - 1             # number of weight layers

    # === Output layer ===
    A_prev, Z = cache.pop()         # A_prev: (m, n_{L-1}), Z: (m, n_L)
    delta = backward_loss(output, label)             # (m, n_L)
    delta = self.backward_g(Z, delta)                # element-wise activation derivative

    # Gradients for layer L
    dW[L] = (A_prev.T @ delta) / m        # (n_{L-1}, m) @ (m, n_L) -> (n_{L-1}, n_L), averaged
    db[L] = np.mean(delta, axis=0, keepdims=True)  # (1, n_L)

    # === Hidden layers ===
    for i in reversed(range(1, L)):
        A_prev, Z = cache.pop()                    # A_prev: (m, n_{i-1}), Z: (m, n_i)
        # propagate delta backward through weights and activation
        delta = delta @ self.W[i + 1].T            # (m, n_{i+1}) @ (n_{i+1}, n_i) -> (m, n_i)
        delta = self.backward_f(Z, delta)          # apply f'(Z)

        # accumulate gradients, averaged over batch
        dW[i] = (A_prev.T @ delta) / m              # (n_{i-1}, n_i)
        db[i] = np.mean(delta, axis=0, keepdims=True)  # (1, n_i)

    return dW, db

    

#allow vectorisation  
# input 1d array or list internaly convert to 2d array (1 x n) for matrix multiplication
# function for a single forward step 
# see how i handle timeout delay











#     def update(
#         self,
#         dW: dict[int, np.ndarray],
#         db: dict[int, np.ndarray],
#         learning_rate: float = 0.01,
#         lambda_reg: float = 0.01,
#         timeout_delay: float = 1.0,
#     ) -> None:
#         for i in dW:
#             self.W[i] -= learning_rate * (dW[i] + 2 * lambda_reg * self.W[i])
#             self.b[i] -= learning_rate * db[i]
#         self._clip_weights_and_biases()
#         self.set_weights_and_biases(timeout_delay)

#     def train(
#         self,
#         data: np.ndarray,
#         labels: np.ndarray,
#         epochs: int,
#         learning_rate: float = 0.01,
#         learning_rate_optimiser: learning_rate_optimizer_type = lambda lr, epoch, dW, db: lr,
#         loss: loss_function_type = (
#             lambda x, y: np.mean((x - y) ** 2),
#             lambda x, y: 2 * (x - y) / np.size(y),
#         ),
#         lambda_reg: float = 0.0,
#         batch_size: Optional[int] = None,
#         saving: bool = False,
#         save_step: int = 1,
#         saving_improvement: float = 0.8,
#         err_min_init: float = 0.001,
#         printing: bool = True,
#         print_step: int = 10,
#         graphing: bool = True,
#     ) -> None:
#         samples = len(data)
#         batch_size = batch_size or samples
#         error_history = []
#         err_min = err_min_init
#         subdirectory = ""

#         if saving:
#             directory = self.name + "_training"
#             os.makedirs(directory, exist_ok=True)
#             subdirectory = os.path.join(directory, str(np.datetime64('now')))
#             os.makedirs(subdirectory, exist_ok=True)

#         for epoch in range(epochs):
#             err = 0
#             indices = np.random.permutation(samples)
#             data, labels = data[indices], labels[indices]

#             dW, db = self._empty_grad()
#             for i in range(0, samples, batch_size):
#                 x_batch = data[i:i + batch_size]
#                 y_batch = labels[i:i + batch_size]

#                 for x, y in zip(x_batch, y_batch):
#                     output, cache = self._forward_propagation(x)
#                     err += loss[0](output, y)
#                     dW, db = self._add_grad(dW, db, self._backward_propagation(output, y, cache, loss[1]))

#                 dW, db = self._divide_grad(dW, db, batch_size)
#                 learning_rate = learning_rate_optimiser(learning_rate, epoch, dW, db)
#                 self.update(dW, db, learning_rate, lambda_reg)

#             err /= samples
#             if graphing:
#                 error_history.append(err)

#             if printing and (epoch + 1) % print_step == 0:
#                 print(f"Epoch {epoch + 1}/{epochs} - Error: {err:.6e}")

#             if saving and (epoch + 1) % save_step == 0 and err < err_min:
#                 err_min = err * saving_improvement
#                 np.savez(os.path.join(subdirectory, f"{self.name}_{err:.6e}.npz"), W=self.W, b=self.b)

#         if graphing:
#             plt.plot(range(1, epochs + 1), error_history, label="Training Error")
#             plt.xlabel("Epoch")
#             plt.ylabel("Error")
#             plt.title("Training Error Over Time")
#             plt.legend()
#             plt.grid()
#             plt.show()

# # Vectorized forward locally, before sending to Arduino
# # Xbatch: (m, n_input)
# Zs, As = [], []
# A = Xbatch  # shape (m, n0)
# for i in range(1, L):
#     Z = A @ self.W[i] + self.b[i]     # Z: (m, n_i)
#     A = self.f(Z)                     # A: (m, n_i)
#     Zs.append(Z); As.append(A)

# # Now you could send the **entire** batch of A as JSON:
# resp = interface.send_and_receive({
#     "cmd":"forward_batch",
#     "inputs": A.tolist()              # list-of-lists, length m
# }, expected_keys=["outputs"])

# Yhat = np.array(resp["outputs"])      # shape (m, n_L)

# err_batch = np.mean((Yhat - Ybatch)**2)     # vectorized
# # Compute delta at output layer:
# delta_L = 2*(Yhat - Ybatch)/m * g_prime(Zs[-1])  # (m, n_L)
# # And so on, using matrix ops:
# dW[L] = As[-2].T @ delta_L                   # (n_{L-1}, n_L)
# db[L] = delta_L.sum(axis=0, keepdims=True)   # (1, n_L)
