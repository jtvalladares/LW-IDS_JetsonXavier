import socket
import pickle
import time
import struct
import csv
import numpy as np

class JetsonXavierServer: 

    def __init__(self, host, port, storage_file):
        self.host = host
        self.port = port
        self.server_socket = None
        self.conn = None
        self.addr = None

        self.model = None

        # Data storage
        self.fieldnames = ["pred_time", "pred"]
        self.csv_file = storage_file

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        print(f"Server listening in {self.host}:{self.port}")
        self.conn, self.addr = self.server_socket.accept()
        self.conn.settimeout(10)
        print(f"Connection made with {self.addr}")

    def close(self):
        if self.conn:
            self.conn.close()
        if self.server_socket:
            self.server_socket.close()
        print("Connection closed.")

    def load_model(self, model_path):
        # Load the decision tree model
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
            print(f"Loaded: {model_path}")
        
    def recvall(self, n):
        """Helper function to receive exactly n bytes."""
        data = bytearray()
        while len(data) < n:
            packet = self.conn.recv(n - len(data))
            if not packet:
                return None  # Connection closed
            data.extend(packet)
        return data

    def rx_process_data(self):
        try: 

            raw_length = self.recvall(4)
            if not raw_length:
                print(">>> [Rx] No data length received.")
                return None
            
            bytes_length = int.from_bytes(raw_length, 'big')
            packet_raw = self.recvall(bytes_length)
            if not packet_raw:
                print(">>> [Rx] No data packet received.")
                return None
            
            # Deserialize the data
            data_row = pickle.loads(packet_raw)
            print(f">>> [Rx] Received data: {data_row}")
            return np.array(data_row)
        
        except (socket.timeout, ConnectionResetError, EOFError, pickle.UnpicklingError) as e:
            print(f">>> [Rx] Error receiving data: {e} <<<")
            return None
        
    def tx_process_data(self, data):
        try:
            # Serialize the data
            serialized_data = pickle.dumps(data)
            msg = struct.pack('>I', len(serialized_data)) + serialized_data
            print(f">>> [Tx] Serialized data size: {len(serialized_data)} bytes")
            print(f">>> [Tx] Sent data: {data}")
            self.conn.sendall(msg)  
        except (socket.timeout, ConnectionResetError, EOFError) as e:
            print(f">>> [Tx] Error sending data: {e} <<<")


    def handle_client(self):
        
        # Esperar a recibir el nombre del archivo antes de procesar datos
        first_msg = self.rx_process_data()

        if isinstance(first_msg, dict) and "file_name" in first_msg:
            file_name = first_msg["file_name"]
            print(f">>> [Rx] File name received from client: {file_name}")
        else:
            print(">>> [Rx] Expected file name not received. Exiting.")
            return

        with open(self.csv_file, "w") as file:
            csv_writer = csv.DictWriter(file, self.fieldnames)
            csv_writer.writeheader()

        while True:
            
            row = self.rx_process_data()

            if row is None:
                print(">>> [Rx] No more data. Closing connection.")
                break

            if not isinstance(row, (np.ndarray, list)):
                print(">>> [Rx] Invalid data type. Skipping.")
                continue

            # print(f" >>> [Rx] Row to run model prediction: {row}")
            data_row = row.reshape(1,-1)

            start_time = time.time()
            prediction = self.model.predict(data_row)
            end_time = time.time()

            pred_value = prediction[0] if hasattr(prediction, '__getitem__') else prediction
            print(f" >>> [Rx] Prediction: {prediction}")
            print(f" >>> [Rx] Prediction time: {end_time - start_time:.4f} seconds")

            self.tx_process_data(pred_value)

            time_pred = end_time - start_time
            data = [time_pred, pred_value]

            self.write_data_file(data)

    def write_data_file(self, data):
        with open(self.csv_file, "a") as file:
            csv_writer = csv.DictWriter(file, self.fieldnames)
            info = {
                self.fieldnames[0]: data[0],
                self.fieldnames[1]: data[1]
            }
            csv_writer.writerow(info)