import ctypes
import time
from threading import Thread
from pymodbus.datastore import ModbusSlaveContext, ModbusSequentialDataBlock, ModbusServerContext
from pymodbus.server import StartTcpServer


class ModbusSlaveHandler:
    def __init__(self, ip="0.0.0.0", port=5020):
        """
        Initialize a Modbus TCP slave (server) context and start it in a background thread.

        Args:
            ip (str): IP address to bind the Modbus server. Default is "0.0.0.0" (all interfaces).
            port (int): TCP port to listen on. Default is 5020.
        """
        # Create Modbus data block with 30 holding registers (starting from address 0)
        self.store = ModbusSlaveContext(
            hr=ModbusSequentialDataBlock(0, [0] * 30)
        )
        self.context = ModbusServerContext(slaves=self.store, single=True)

        # Start Modbus TCP server in a background daemon thread
        self.server_thread = Thread(target=self.start_server, args=(ip, port), daemon=True)
        self.server_thread.start()

    def start_server(self, ip, port):
        """
        Start the Modbus TCP server.

        Args:
            ip (str): IP address to bind.
            port (int): Port to listen on.
        """
        print(f"[Slave] Starting Modbus TCP Server at {ip}:{port}")
        StartTcpServer(context=self.context, address=(ip, port))

    def _run_read_and_write(self, action):
        """
        Handle read-write communication with the Modbus master (simulated protocol).

        Args:
            action (list): List of 10 integers representing action parameters.

        Returns:
            tuple: (_, _, _, total_time, final_weight)
        """
        if len(action) != 10:
            raise ValueError("Action list must contain exactly 10 elements")

        # Reset old status flags
        self.context[0x00].setValues(3, 10, [0])  # Reset status flag (register 10)
        self.context[0x00].setValues(3, 17, [0])  # Reset result flag (register 17)

        # Write action parameters to registers starting at address 0
        self.context[0x00].setValues(3, 0, action)
        print(f"[Slave] Parameters written: {action}")
        self.context[0x00].setValues(3, 10, [1])  # Notify master: parameters written
        print(f"[Slave] Parameters written, waiting for master to read...")

        # Wait for master to confirm read by setting status to 2
        while True:
            status = self.context[0x00].getValues(3, 10, count=1)[0]
            if status == 2:
                print("[Slave] Master has read parameters, clearing status")
                self.context[0x00].setValues(3, 10, [0])  # Clear status
                break
            time.sleep(0.05)

        # Wait for result data from the master (writing to registers 15–18)
        print("[Slave] Waiting for result from master...")
        while True:
            flag = self.context[0x00].getValues(3, 19, count=1)[0]
            if flag == 1:
                result = self.context[0x00].getValues(3, 15, count=4)
                total_time = self.convert_to_signed_32bit(result[0], result[1])
                final_weight = self.convert_to_signed_32bit(result[2], result[3])
                self.context[0x00].setValues(3, 19, [0])  # Clear result flag
                print(f"[Slave] Result received: time={total_time}, weight={final_weight}")
                return None, None, None, total_time, final_weight
            time.sleep(0.05)

    def convert_to_signed_32bit(self, high, low):
        """
        Convert two 16-bit registers into one signed 32-bit integer.

        Args:
            high (int): High-order 16-bit word.
            low (int): Low-order 16-bit word.

        Returns:
            int: Signed 32-bit integer.
        """
        value_32bit = (low << 16) | high  # Combine into a 32-bit unsigned integer
        signed_value = ctypes.c_int32(value_32bit).value  # Cast to signed 32-bit int
        return signed_value


# Instantiate a Modbus slave server listening on a specific IP and port
modbus_slave_client = ModbusSlaveHandler(ip="192.168.31.54", port=5020)

