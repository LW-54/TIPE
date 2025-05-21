import serial
import serial.tools.list_ports
import threading
import time
from typing import Tuple, Any
from os import ttyname
from pty import openpty


def serial_innit(port: int, baud: int = 9600) -> serial.Serial:
    serialInst = serial.Serial()
    serialInst.baudrate = baud
    serialInst.port = port
    serialInst.open()
    return serialInst


def select_port() -> int :
    ports = serial.tools.list_ports.comports()
    portsList = []
    for i, p in enumerate(ports):
        portsList.append(str(p).split()[0])
        print(f"{i} - {p}")
    if portsList:
        return portsList[int(input("Select port n° : "))]
    else :
        raise Exception("No ports available")


def select_port_innit(baud: int = 9600) -> serial.Serial:
    return serial_innit(select_port(),baud)



def serial_read(serialInst: serial.Serial, timeout: float = 0.1) -> str | None:
    try: #think about error handling should it crash or return none or other?
        serialInst.timeout = timeout  # look how it works is it global ? is there a need for a await function with explicit timeout handling?
        return serialInst.readline().decode('utf-8').strip() #how does it read ? what caracter does it stop at etc ,
    except Exception as e:
        print(f"Error reading from serial: {e}")
        return None
    

def serial_write(serialInst: serial.Serial, data: Any) -> None: # think about the typing of data is it Any or string and it is specified when calling
    try:
        serialInst.write(data.encode('utf-8')) #what does it do to already encoded data ? does it add a newline caracter ?
    except Exception as e: # error handling
        print(f"Error writing to serial: {e}")



def serial_monitor(serialInst: serial.Serial) -> None:

    exit_flag = False

    def serial_read_thread(serialInst: serial.Serial):
        nonlocal exit_flag # is it needed herere not here  before
        while not exit_flag:
            incoming_data = serial_read(serialInst)
            if incoming_data:
                print(f"\r| {incoming_data}", end="\n> ", flush=True)

    def user_input_thread(serialInst: serial.Serial):
        nonlocal exit_flag
        while not exit_flag:
            user_input = input("> ")
            if user_input.lower() == 'exit':
                print("\nExiting serial monitor...")
                exit_flag = True
            serialInst.write(user_input.encode('utf-8'))

    serial_thread = threading.Thread(target=serial_read_thread, args=(serialInst,))
    user_thread = threading.Thread(target=user_input_thread, args=(serialInst,))

    serial_thread.start()
    user_thread.start()

    serial_thread.join()
    user_thread.join()



def sim_serial(baud: int = 9600) -> Tuple[int, int]:
    master, slave = openpty()
    return  master, ttyname(slave)


def sim_serial_innit(baud: int = 9600) -> Tuple[serial.Serial, int]:
    master, slave = sim_serial(baud)
    return master, serial_innit(slave)


def sim_serial_read(port, timeout) :
    pass  
    #         command = b""
    #         while not command.endswith(b"\n"):
    #             command += os.read(port, 1)
    #         command = command.decode("utf-8").strip()


def sim_serial_write(port, data) :
    pass # os.write(port, f"Analog value on O{index}: {analog_value}\n".encode())


def run_sim(sim , master): #do the typing expects a function
    thread = threading.Thread(target=sim, args=[master], daemon=True)
    thread.start()


def foo():
    return 54

def main():
    ser = serial_innit()
    print(ser)


if __name__ == "__main__":
    main()

# test specefic cases and random data 