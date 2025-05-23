import socket
import pickle
import numpy as np 
import time
import struct

class JetsonNanoClient: 

    def __init__(self, host, port, storage_file, num_regs):
        self.host = host
        self.port = port

        self.client_socket = None

        self.test_path = None

        # Data for each test
        self.num_regs = num_regs
        self.idx = 0
        self.predictions = (-1) * np.ones(shape=(self.num_regs,))
        self.times = (-1) * np.ones(shape=(self.num_regs,))
        self.sent_records = 0
        self.received_records = 0

        self.times_txt_file = storage_file

    def connect(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))
        print("Connected to server. Processing operations.")

    def close(self):
        if self.client_socket:
            self.client_socket.close()
        print("Connection closed.")

    def load_test_file(self, test_path):
        # Load the test files
        self.test_path = test_path
        print(f"Loaded: {self.test_path}")

    def recvall(self, n):
        """Helper function to receive exactly n bytes."""
        data = bytearray()
        while len(data) < n:
            packet = self.client_socket.recv(n - len(data))
            if not packet:
                return None
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
                print(">>> [Rx] No data packer received.")
                return None
            
            # Deserialize the data
            pred = pickle.loads(packet_raw)
            #print(f" >>> [Rx] Received prediction: {pred}")
            return pred

        except (socket.timeout, ConnectionResetError, EOFError, pickle.UnpicklingError) as e:
            print(f"Error receiving data: {e}")
            return None
        
    def tx_process_data(self, data):
        try:
            # Serialize the data
            serialized_data = pickle.dumps(data)
            msg = struct.pack('>I', len(serialized_data)) + serialized_data
            #print(f">>> [Tx] Sending data: {data}")
            self.client_socket.sendall(msg)
            #print(">>> [Tx] Data sent successfully.")
        except (socket.timeout, ConnectionResetError, EOFError, pickle.PicklingError) as e:
            print(f">>> [Tx] Error sending data: {e} <<<")

    def tx_rx(self):
        
        data = np.load(self.test_path)
        split_file = self.test_path.split('/')
        split_file = split_file[-1].split('.')
        exp_num = split_file[0][-1]
        test_num = split_file[1][0]
        class_num = split_file[2][9]
        exp_params = [exp_num, test_num, class_num]
        #print(f"Experiment parameters: {exp_params}")

        real_labels = data[:,-2]
        other_predictions = data[:,-1]
        # Delete last two columns
        data = data[:, :-2] 

        for reg in data:
            
            start_time = time.time()

            # Sent model index & data
            self.tx_process_data(reg)
            self.sent_records += 1

            # Receive prediction
            prediction = self.rx_process_data()
            end_time = time.time()

            self.predictions[self.idx] = prediction
            self.times[self.idx] = end_time - start_time
            # print(f">>> [Rx] Prediction: {prediction}, Time taken: {end_time - start_time:.4f} seconds")
            self.idx += 1

            self.received_records += 1

        pred_accuracy, other_accuracy = self.compute_metrics(real_labels, self.predictions, other_predictions)

        print(f"--- Mean Time: {np.mean(self.times):.4f} seconds ---")
        print(f"--- Time Deviation: {np.std(self.times):.4f} seconds ---")
        print(f"--- Sent Records: {self.sent_records} ---")
        print(f"--- Received Records: {self.received_records} ---")

        self.log_times_to_txt(pred_accuracy, other_accuracy, exp_params)
        self.close()
                    
    def compute_metrics(self, real_labels, predictions, other_predictions):
        # Calculate accuracy
        prediction_accuracy = sum(1 for p, r in zip(predictions, real_labels) if p == r) / len(real_labels)
        other_accuracy = sum(1 for p, o in zip(predictions, other_predictions) if p == o) / len(other_predictions)
        print(f"--- Prediction accuracy: {prediction_accuracy * 100:.2f} ---%")
        print(f"--- Other system accuracy: {other_accuracy * 100:.2f}% ---")
        return prediction_accuracy, other_accuracy
    
    def log_times_to_txt(self, pred_accuracy, other_accuarcy, exp_params):
        with open(self.times_txt_file.format(exp_params[0], exp_params[1], exp_params[2]), 'w') as f:
            for time_value in self.times:
                f.write(f"{time_value}\n")
            f.write(f"--- Prediction accuracy: {pred_accuracy * 100:.2f}% ---\n")
            f.write(f"--- Other system accuracy: {other_accuarcy * 100:.2f}% ---\n")
        print(f"Times logged to {self.times_txt_file.format(exp_params[0], exp_params[1], exp_params[2])}")