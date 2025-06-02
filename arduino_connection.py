import serial.tools.list_ports

from encephalon.serial_utils import SerialJSONInterface, SerialCommunicationError, handshake

# Update this to your Arduino's actual port!


def test_connection():
    try:
        ports = serial.tools.list_ports.comports()
        portsList = []
        for i, p in enumerate(ports):
            portsList.append(str(p).split()[0])
            print(f"{i} - {p}")
        if not portsList:
            raise Exception("No ports available")
        PORT = portsList[int(input("Select port n° : "))]

        print(f"Connecting to {PORT}...")
        interface = SerialJSONInterface(port=PORT, baud=9600, timeout=1.0)
        handshake(interface)

        # Example: set a known weight
        print("Sending SET command...")
        set_response = interface.send_and_receive({
            "cmd": "set",
            "param": "W1:0:0",
            "value": 0.42
        }, expected_keys=["status"])
        print("SET response:", set_response)

        # Example: set an input
        print("Setting input...")
        interface.send_and_receive({
            "cmd": "set",
            "param": "I0",
            "value": 1.0
        }, expected_keys=["status"])

        interface.send_and_receive({
            "cmd": "set",
            "param": "I1",
            "value": -1.0
        }, expected_keys=["status"])

        # Example: read output
        print("Requesting forward output...")
        read_response = interface.send_and_receive({
            "cmd": "read",
            "index": 0
        }, expected_keys=["value"])
        print("Output:", read_response["value"])

        interface.close()
        print("Test complete.")

    except SerialCommunicationError as e:
        print(f"[Error] {e}")

if __name__ == "__main__":
    test_connection()
