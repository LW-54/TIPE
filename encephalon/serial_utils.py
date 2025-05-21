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
