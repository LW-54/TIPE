import numpy as np

from .serial_utils import SerialJSONInterface


class Grapher:
    def __init__(
        self,
        serial_interface: SerialJSONInterface,
        verbose: bool = False,
    ) -> None:
        
        if not isinstance(serial_interface, SerialJSONInterface):
            raise TypeError("serial_interface must be a SerialJSONInterface instance.")

        self.serial = serial_interface
        self.verbose = verbose

        self._log("Grapher initialised")
    

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.name}] {msg}")

    def graph(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, ndmin=2)
        return np.array((self.serial.send_and_receive({
            "cmd":"graph",
            "input": X.tolist()         
        }, expected_keys=["output"]))["output"], ndmin=2)


    def set(self, pin : int, v : float) -> None:

        if not 0<=v<=5 :
            raise ValueError(f"Input value {v} out of range")

        response = self.serial.send_and_receive({
            "cmd":"set",
            "param": f"I{pin}",
            "value": v         
        })
    
    def read(self, pin : int) -> float :
        return float(self.serial.send_and_receive({
            "cmd":"read",
            "index": pin,        
        }, expected_keys=["value"])["value"])
