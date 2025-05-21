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
        interface.send_and_receive({"cmd": "X"}, expected_keys=["expected"])

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

    interface, sim = simulate_serial(no_response, timeout=0.1)
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
