
import json
import os
from typing import Callable, Optional, Union
from datetime import datetime
import warnings

import matplotlib.pyplot as plt
import numpy as np

from .serial_utils import SerialJSONInterface, SerialProtocolError, SerialTimeoutError
from .types_and_functions import (
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
        
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise ValueError("All layer sizes must be positive integers.")
        if not isinstance(serial_interface, SerialJSONInterface):
            raise TypeError("serial_interface must be a SerialJSONInterface instance.")
        if not callable(f[0]) or not callable(f[1]) or not callable(g[0]) or not callable(g[1]):
            raise TypeError("Activation functions f and g must be (function, derivative) tuples.")

        self.layers = layers
        if len(self.layers) < 2:
            raise ValueError("Network must have at least 2 layers (input and output).")

        self.serial = serial_interface
        self.name = name
        self.verbose = verbose

        # Activation functions
        self.f, self.backward_f = f
        self.g, self.backward_g = g

        # Weights and biases: layer i connects layers[i-1] to layers[i]
        self.W = {
            i: np.random.randn(self.layers[i - 1], self.layers[i]) * np.sqrt(2 / self.layers[i - 1])
            for i in range(1, len(self.layers))
        }
        self.b = {i: np.zeros((1, self.layers[i])) for i in range(1, len(self.layers))}

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
            "W": {str(i): v.tolist() for i, v in self.W.items()},
            "b": {str(k): v.tolist() for k, v in self.b.items()},
        }
        response = self.serial.send_and_receive(payload, expected_keys=["status"])
        if response["status"] != "OK":
            raise SerialProtocolError(f"Arduino failed to load weights and biases: {response}")

    def use(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, ndmin=2)

        if X.shape[1] != self.layers[0]:
            raise ValueError(f"Input dimension {X.shape[1]} does not match network input size {self.layers[0]}")

        response = self.serial.send_and_receive({
            "cmd":"forward",
            "input": X.tolist()         
        }, expected_keys=["output"])

        return self.g(np.array(response["output"], ndmin=2))

    def _forward_propagation(self, X: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        cache = []
        A = np.array(X, ndmin=2)
        for i in self.W:
            Z = A @ self.W[i] + self.b[i]
            cache.append((A, Z))
            A = self.f(Z)
        return self.use(X), cache

    def _backward_propagation(
        self,
        output: np.ndarray,    # shape (m, n_L)
        label: np.ndarray,     # shape (m, n_L)
        cache: list[tuple[np.ndarray, np.ndarray]],  # [(A⁽⁰⁾, Z⁽¹⁾), …, (A⁽L⁻¹⁾, Z⁽L⁾)]
        backward_loss: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:

        m = output.shape[0]             # batch size
        L = len(self.layers) - 1        # number of weight layers

        dW = {i: np.zeros_like(self.W[i]) for i in self.W}
        db = {i: np.zeros_like(self.b[i]) for i in self.b}

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

    def _update_weights_and_biases(
        self,
        dW: dict[int, np.ndarray],
        db: dict[int, np.ndarray],
        learning_rate: float,
        lambda_reg: float,
    ) -> None:
        for i in dW:
            self.W[i] -= learning_rate * (dW[i] + 2 * lambda_reg * self.W[i])
            self.b[i] -= learning_rate * db[i]
        self._clip_weights_and_biases()
        self.set_weights_and_biases() 

    def save(self, filepath: str) -> None:
        payload = {
            "metadata": {
                "name": self.name,
                "timestamp": datetime.now().isoformat() + "Z",
                "layers": self.layers,
            },
            "W": {str(i): self.W[i].tolist() for i in self.W},
            "b": {str(i): self.b[i].tolist() for i in self.b},
        }
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)
        self._log(f"[{self.name}] Saved to JSON: {filepath}")

    def load(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            payload = json.load(f)

        # (Re)build internal structures
        self.name = payload["metadata"].get("name", self.name)
        self.layers = payload["metadata"]["layers"]
        self.W = {int(i): np.array(payload["W"][i], ndmin=2) for i in payload["W"]}
        self.b = {int(i): np.array(payload["b"][i], ndmin=2) for i in payload["b"]}

        self._log(f"[{self.name}] Loaded from JSON: {filepath}")

    def train(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        epochs: int,
        learning_rate: float = 0.01,
        learning_rate_optimiser: learning_rate_optimizer_type = fixed_learning_rate(),
        loss_function: loss_function_type = mse,
        lambda_reg: float = 0.0,
        batch_size: int = 1,
        saving: bool = False,
        save_step: int = 1,
        saving_improvement: float = 0.8,
        loss_min_init: float = 0.001,
        verbose: Optional[bool] = None,
        print_step: int = 10,
        graphing: bool = True,
    ) -> None:
        if verbose:
            self.verbose = verbose

        data, labels = np.array(data, ndmin=2), np.array(labels, ndmin=2)

        samples, n_in = data.shape
        m, n_out = labels.shape

        if m != samples:
            raise ValueError(f"Data has {samples} samples but labels has {m}")
        if n_out != self.layers[-1]:
            raise ValueError(
                f"Label dimension ({n_out}) must match network output size ({self.layers[-1]})"
            )
        if epochs < 1:
            raise ValueError("epochs must be ≥ 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if batch_size < 1:
            warnings.warn("batch_size < 1, using batch_size = 1")
            batch_size = 1
        if batch_size > samples:
            warnings.warn("batch_size > num_samples, using full batch")
            batch_size = samples
        if not callable(learning_rate_optimiser):
            raise TypeError("learning_rate_optimiser must be a callable")
        if (not isinstance(loss_function, tuple)
            or len(loss_function) != 2
            or not callable(loss_function[0])
            or not callable(loss_function[1])):
            raise TypeError("loss_function must be (loss, loss_derivative) tuple of callables")
        
        if graphing:
            loss_history = []
            plt.ion()
            fig, ax = plt.subplots()
            line, = ax.plot([], [], label="Training Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"{self.name} Training Loss over Epochs")
            ax.legend()
            ax.grid(True)
        
        if saving:
            loss_min = loss_min_init
            directory = self.name + "_training"
            os.makedirs(directory, exist_ok=True)
            subdirectory = os.path.join(directory, str(np.datetime64('now')))
            os.makedirs(subdirectory, exist_ok=True)

        for epoch in range(epochs):
            loss = 0
            indices = np.random.permutation(samples)
            data, labels = data[indices], labels[indices]

            for i in range(0, samples, batch_size):
                x_batch = data[i:i + batch_size]
                y_batch = labels[i:i + batch_size]

                output, cache = self._forward_propagation(x_batch)
                loss += loss_function[0](output, y_batch)
                dW, db = self._backward_propagation(output, y_batch, cache, loss_function[1])
                learning_rate = learning_rate_optimiser(learning_rate, epoch, dW, db)
                self._update_weights_and_biases(dW, db, learning_rate, lambda_reg)

            loss /= samples

            if graphing:
                loss_history.append(loss)
                # update data
                line.set_data(range(1, epoch + 1), loss_history)
                # adapt axes
                ax.relim()
                ax.autoscale_view()
                # redraw and pause briefly
                fig.canvas.draw()
                plt.pause(0.001)

            if (epoch + 1) % print_step == 0:
                self._log(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6e}")

            if saving and (epoch + 1) % save_step == 0 and loss < loss_min:
                loss_min = loss * saving_improvement
                self.save(os.path.join(subdirectory, f"{self.name}_{loss:.6e}.json"))

        if graphing:
            plt.ioff()
            plt.show()
