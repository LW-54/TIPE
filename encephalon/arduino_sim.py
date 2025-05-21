
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
