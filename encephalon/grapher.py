
import numpy as np

from .serial_utils import SerialJSONInterface


class Grapher:
    """A class for graphing data from the Arduino.

    This class provides a high-level interface for sending data to the Arduino
    and reading the results for graphing.
    """
    def __init__(
        self,
        serial_interface: SerialJSONInterface,
        verbose: bool = False,
    ) -> None:
        """Initializes the Grapher.

        Args:
            serial_interface: An instance of SerialJSONInterface for communicating
                with the Arduino.
            verbose: If True, print verbose output.
        """
        
        if not isinstance(serial_interface, SerialJSONInterface):
            raise TypeError("serial_interface must be a SerialJSONInterface instance.")

        self.serial = serial_interface
        self.verbose = verbose

        self._log("Grapher initialised")
    

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Grapher] {msg}")

    # def graph(self, X: np.ndarray) -> np.ndarray: #il faudrer trouver le bug
    #     X = np.array(X, ndmin=2)
    #     return np.array((self.serial.send_and_receive({
    #         "cmd":"graph",
    #         "input": X.tolist()         
    #     }, expected_keys=["output"]))["output"], ndmin=2)

    def graph(self, X: np.ndarray, out : int = 0) -> np.ndarray:
        """Graphs the given data.

        Args:
            X: The data to graph.
            out: The output pin to read from.

        Returns:
            The data read from the Arduino.
        """
        X = np.array(X, ndmin=2)
        Y = []
        n,m = X.shape
        for i in range(n):
            for j in range(m):
                self.set(j,X[i][j])
            Y.append(self.read(out))
            self._log(f"{i+1}/{n}")
        return np.array(Y, ndmin=2) # dim not realy necessary


    def set(self, pin : int, v : float) -> None:
        """Sets the voltage of a given pin.

        Args:
            pin: The pin to set.
            v: The voltage to set the pin to.
        """

        if not 0<=v<=5 :
            raise ValueError(f"Input value {v} out of range")

        response = self.serial.send_and_receive({
            "cmd":"set",
            "param": f"I{pin}",
            "value": v         
        })
    
    def read(self, pin : int) -> float :
        """Reads the voltage of a given pin.

        Args:
            pin: The pin to read from.

        Returns:
            The voltage of the pin.
        """
        return float(self.serial.send_and_receive({
            "cmd":"read",
            "index": pin,        
        }, expected_keys=["value"])["value"])

