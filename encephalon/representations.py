# functions used to graph and show NN in all sorts of ways

here is my current code write tests covering most of the code using pytest: 

```types_and_functions.py

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


```

```nn.py

import json
import os
from typing import Callable, Optional, Union
from datetime import datetime
import warnings

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
        return self.g(np.array(response["output"]), ndmin=2)

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

```

```arduino_sim.py


import json
from typing import Callable
import numpy as np
from types_and_functions import activation_function_type, ReLU, Id, mse

#add possibility to set and read specefic pins

def make_arduino_simulator(
    layers: list[int],
    f: activation_function_type = ReLU,
    noise_amplitude: float = 0.0
) -> Callable[[dict], dict]:

    fwd_f, back_f = f

    # initialize weights & biases
    W = {
        i: np.random.randn(layers[i - 1], layers[i]) * np.sqrt(2 / layers[i - 1])
        for i in range(1, len(layers))
    }
    b = {i: np.zeros((1, layers[i])) for i in range(1, len(layers))}
    I = np.zeros((1, layers[0])) 
    N = np.random.rand(layers[-1]) * noise_amplitude

    def forward(X: np.ndarray) -> np.ndarray:
        A = X
        for i in range(1, len(layers) - 1):
            A = fwd_f(A @ W[i] + b[i] )
        return A @ W[len(layers) - 1] + b[len(layers) - 1] + N


    def callback(msg: dict) -> dict:
        cmd = msg.get("cmd", "")
        try:
            if cmd == "set_weights_and_biases":
                W_in = msg["W"]
                b_in = msg["b"]
                for k, v in W_in.items():
                    W[int(k)] = np.array(v)
                for k, v in b_in.items():
                    b[int(k)] = np.array(v)
                return {"status": "OK"}

            elif cmd == "forward":
                X = np.array(msg.get("input"), dtype=float, ndmin=2)

                if X.shape[1] != layers[0]:
                    return {"error": f"Expected input width {layers[0]}, got {X.shape[1]}"}

                return {"output": forward(X).tolist()}
            
            elif cmd == "set":
                # msg["param"] like "W1:2:3" or "b2:0" or "I0"
                # msg["value"]: new float
                param = msg["param"]
                value = float(msg["value"])
                if param.startswith("W"):
                    # W<layer>:<row>:<col>
                    layer, row, col = map(int, param[1:].split(":"))
                    W[layer][row, col] = value
                elif param.startswith("b"):
                    # b<layer>:<col>
                    layer, col = map(int, param[1:].split(":"))
                    b[layer][0, col] = value
                elif param.startswith("I"):
                    # I<index>
                    idx = int(param[1:])
                    I[0, idx] = value
                else:
                    return {"error": f"Unknown param '{param}'"}
                return {"status": "OK"}

            elif cmd == "read":
                # msg["index"]: integer output index to read
                idx = int(msg["index"])
                Y = forward(I)
                if idx < 0 or idx >= Y.shape[1]:
                    return {"error": f"Index {idx} out of range"}
                return {"value": float(Y[0, idx])}

            else:
                return {"error": f"Unknown command '{cmd}'"}
            
        except Exception as e:
            # catch any bug in simulator
            return {"error": str(e)}

    return callback

```

```serial_utils.py
import json
import threading
import time
import os
import pty
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import serial


class SerialCommunicationError(Exception):
    """Base exception for serial communication errors."""
    pass


class SerialTimeoutError(SerialCommunicationError):
    """Exception raised when a serial read operation times out."""
    pass


class SerialParseError(SerialCommunicationError):
    """Exception raised when incoming data cannot be parsed as JSON."""
    pass


class SerialProtocolError(SerialCommunicationError):
    """Exception raised when a JSON response is missing expected fields."""
    pass


class SerialWriteError(SerialCommunicationError):
    """Exception raised when writing to the serial port fails."""
    pass


class SerialJSONInterface:
    """
    Manages sending and receiving newline-delimited JSON messages over a serial port.

    Example usage:
        interface = SerialJSONInterface(port='/dev/ttyUSB0', baud=9600, timeout=1.0)
        response = interface.send_and_receive({"cmd": "SET", "value": 42},
                                              expected_keys=["status", "value"])
        interface.close()

    Attributes:
        serial: The underlying pySerial Serial instance.
    """
    def __init__(self,
                 port: Optional[Union[str, int]] = None,
                 baud: int = 9600,
                 timeout: float = 0.5,
                 serial_inst: Optional[Any] = None):
        """
        Initialize the serial interface.

        Args:
            port: The serial port (device name or COM port) to open.
            baud: The baud rate for serial communication.
            timeout: Read timeout in seconds.
            serial_inst: An existing serial.Serial instance (for testing/simulation).

        Raises:
            SerialCommunicationError: If the port cannot be opened.
        """
        if serial_inst is not None:
            self.serial = serial_inst
        elif port is not None:
            try:
                self.serial = serial.Serial(port=port, baudrate=baud, timeout=timeout)
            except Exception as e:
                raise SerialCommunicationError(f"Failed to open serial port {port}: {e}")
        else:
            raise ValueError("Either port or serial_inst must be provided")

    def send_command(self, command: Dict[str, Any]) -> None:
        """
        Send a JSON-serializable command over the serial port. A newline is appended.

        Args:
            command: A dictionary representing the JSON command.

        Raises:
            SerialWriteError: If writing to the port fails.
            SerialCommunicationError: If the command is not JSON-serializable.
        """
        try:
            message = json.dumps(command)
        except (TypeError, ValueError) as e:
            raise SerialCommunicationError(f"Command not JSON serializable: {e}")

        message_bytes = (message + "\n").encode("utf-8")
        try:
            self.serial.write(message_bytes)
        except Exception as e:
            raise SerialWriteError(f"Failed to write to serial port: {e}")

    def receive_response(self, expected_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Read a line from the serial port, parse it as JSON, and validate keys.

        Args:
            expected_keys: List of keys that must be present in the JSON response.

        Returns:
            The parsed JSON object as a dict.

        Raises:
            SerialTimeoutError: If no data is received before timeout.
            SerialParseError: If the data is not valid JSON.
            SerialProtocolError: If the JSON is missing expected keys.
        """
        try:
            raw_bytes = self.serial.readline()
        except Exception as e:
            raise SerialTimeoutError(f"Serial read failed or timed out: {e}")

        if not raw_bytes:
            raise SerialTimeoutError("No data received (read timeout)")

        try:
            raw_str = raw_bytes.decode("utf-8").strip()
        except Exception as e:
            raise SerialParseError(f"Failed to decode bytes: {e}")

        if not raw_str:
            raise SerialTimeoutError("Received empty line from serial")

        try:
            data = json.loads(raw_str)
        except json.JSONDecodeError as e:
            raise SerialParseError(f"Received invalid JSON: {e}")

        if expected_keys:
            missing = [key for key in expected_keys if key not in data]
            if missing:
                raise SerialProtocolError(f"Missing keys in response: {missing}")

        return data

    def send_and_receive(self,
                         command: Dict[str, Any],
                         expected_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Send a command and wait for the JSON response.

        Args:
            command: The command to send as a dict.
            expected_keys: Keys expected in the response JSON.

        Returns:
            The parsed JSON response.

        Raises:
            SerialCommunicationError: On send or receive failures.
        """
        self.send_command(command)
        return self.receive_response(expected_keys)

    def close(self) -> None:
        """Close the underlying serial port."""
        try:
            self.serial.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SerialSimulator:
    """
    Simulator that listens on a pseudo-serial master for JSON commands
    and sends back JSON responses via the same interface.
    """
    def __init__(self,
                 master_fd: int,
                 response_callback: Callable[[Dict[str, Any]], Dict[str, Any]],
                 encoding: str = "utf-8"):
        """
        Initialize the SerialSimulator.

        Args:
            master_fd: File descriptor for the pseudo-serial master end.
            response_callback: Function that takes a JSON dict and returns a JSON response dict.
            encoding: Encoding for serial communication.
        """
        self.master_fd = master_fd
        self.response_callback = response_callback
        self.encoding = encoding
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the simulator thread."""
        if self._thread and self._thread.is_alive():
            return  # already running
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the simulator to stop and wait for thread termination."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1)

    def _run(self) -> None:
        """Internal loop: read JSON commands and write responses."""
        buffer = b""
        while not self._stop_event.is_set():
            try:
                byte = os.read(self.master_fd, 1)
            except OSError:
                break
            if not byte:
                time.sleep(0.01)
                continue
            buffer += byte
            if byte == b"\n":
                try:
                    raw = buffer.decode(self.encoding).strip()
                except Exception:
                    buffer = b""
                    continue

                buffer = b""
                if not raw:
                    continue

                try:
                    command = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                try:
                    response = self.response_callback(command)
                except Exception:
                    response = None

                if response is not None:
                    try:
                        out = (json.dumps(response) + "\n").encode(self.encoding)
                        os.write(self.master_fd, out)
                    except Exception:
                        continue


def simulate_serial(response_callback: Callable[[Dict[str, Any]], Dict[str, Any]],
                    baud: int = 9600,
                    timeout: float = 0.5) -> Tuple[SerialJSONInterface, SerialSimulator]:
    """
    Create a pseudo-serial interface and simulator for testing.

    Args:
        response_callback: Function mapping incoming JSON to outgoing JSON.
        baud: Baud rate for the serial interface.
        timeout: Read timeout for the serial interface.

    Returns:
        A tuple (interface, simulator) where:
            interface: SerialJSONInterface connected to the slave end.
            simulator: SerialSimulator listening on the master end.

    Example:
        def echo_callback(msg):
            return {"echo": msg}

        interface, sim = simulate_serial(echo_callback)
        sim.start()
        response = interface.send_and_receive({"cmd": "PING"}, expected_keys=["echo"])
    """
    master_fd, slave_fd = pty.openpty()
    slave_name = os.ttyname(slave_fd)
    try:
        serial_inst = serial.Serial(port=slave_name, baudrate=baud, timeout=timeout)
    except Exception as e:
        raise SerialCommunicationError(f"Failed to open simulated serial port {slave_name}: {e}")
    simulator = SerialSimulator(master_fd, response_callback)
    return SerialJSONInterface(serial_inst=serial_inst), simulator

```

```test_serial.py
import pytest
import json
import os
import time

from encephalon.serial_utils import (
    simulate_serial,
    SerialTimeoutError,
    SerialParseError,
    SerialProtocolError,
    SerialCommunicationError,
)

# Basic echo responder for tests
def echo_callback(msg):
    return {"echo": msg}

def test_successful_echo():
    interface, sim = simulate_serial(echo_callback)
    sim.start()

    response = interface.send_and_receive({"msg": "hello"}, expected_keys=["echo"])
    assert "echo" in response
    assert response["echo"]["msg"] == "hello"

    sim.stop()
    interface.close()


def test_missing_key():
    def incomplete_response(_msg):
        return {"status": "ok"}  # no "expected" key

    interface, sim = simulate_serial(incomplete_response)
    sim.start()

    with pytest.raises(SerialProtocolError):
        interface.send_and_receive({"cmd": "SET"}, expected_keys=["expected"])

    sim.stop()
    interface.close()


def test_invalid_json_response():
    def bad_json_response(_msg):
        # We'll bypass JSON encoding by directly writing invalid bytes
        os.write(sim.master_fd, b"{invalid json}\n")
        return None

    interface, sim = simulate_serial(lambda msg: {})  # will be overridden
    sim.response_callback = bad_json_response
    sim.start()

    with pytest.raises(SerialParseError):
        interface.send_and_receive({"cmd": "ANY"}, expected_keys=[])

    sim.stop()
    interface.close()


def test_timeout_response():
    def no_response(_msg):
        return None  # simulator sends no reply

    interface, sim = simulate_serial(no_response, timeout=0.2)
    sim.start()

    with pytest.raises(SerialTimeoutError):
        interface.send_and_receive({"cmd": "WAIT"}, expected_keys=[])

    sim.stop()
    interface.close()


def test_unserializable_command():
    interface, sim = simulate_serial(echo_callback)
    sim.start()

    with pytest.raises(SerialCommunicationError):
        # Sets are not JSON-serializable
        interface.send_and_receive({"bad": {1, 2, 3}}, expected_keys=[])

    sim.stop()
    interface.close()

```
