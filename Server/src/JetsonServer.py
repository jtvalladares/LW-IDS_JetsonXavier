import socket
import pickle
import time
import struct
import csv
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

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

    def load_models(self, model_path):
        if "E1." in model_path:
            # Load the decision tree model
            with open(model_path + '.pkl', 'rb') as file:
                self.model = pickle.load(file)
        else:
            # Convert and load TFLite model
            h5_path = model_path + '.h5'
            tflite_path = model_path + '.tflite'
            try:
                # Convert only if .tflite doesn't exist yet
                if not os.path.exists(tflite_path):
                    keras_model = load_model(h5_path)
                    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
                    tflite_model = converter.convert()

                    with open(tflite_path, 'wb') as f:
                        f.write(tflite_model)
                    print(f"Model converted to TFLite: {tflite_path}")

                # Load the TFLite model
                self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
                self.interpreter.allocate_tensors()

                # Store input/output details
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()

                print(f"Loaded TFLite model: {tflite_path}")

            except Exception as e:
                print(f"Error loading/converting model: {e}")

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
            #print(f">>> [Rx] Received data: {data_row}")
            return np.array(data_row)
        
        except (socket.timeout, ConnectionResetError, EOFError, pickle.UnpicklingError) as e:
            print(f">>> [Rx] Error receiving data: {e} <<<")
            return None
        
    def tx_process_data(self, data):
        try:
            # Serialize the data
            serialized_data = pickle.dumps(data)
            msg = struct.pack('>I', len(serialized_data)) + serialized_data
            self.conn.sendall(msg)  
        except (socket.timeout, ConnectionResetError, EOFError) as e:
            print(f">>> [Tx] Error sending data: {e} <<<")


    def handle_client(self):
        
        with open(self.csv_file, "w") as file:
            csv_writer = csv.DictWriter(file, self.fieldnames)
            csv_writer.writeheader()

        times = (-1) * np.ones(shape=(117177,))
        idx = 0

        while True:
            row = self.rx_process_data()
            if row is None:
                break

            if 'E1.' in self.csv_file:
                data_row = row.reshape(1,-1)
                start_time = time.time()
                prediction = self.model.predict(data_row)
                end_time = time.time()
            else:
                # TensorFlow Lite prediction
                data_row = row.astype(np.float32)  # TFLite usually needs float32

                expected_shape = self.input_details[0]['shape']
                if len(expected_shape) == 2:
                    data_row = np.expand_dims(data_row, axis=0)
                elif len(expected_shape) == 3:
                    data_row = np.expand_dims(data_row, axis=0)
                    data_row = np.expand_dims(data_row, axis=-1)
                else:
                    raise ValueError(f"Unsupported input shape: {expected_shape}")

                start_time = time.time()
                self.interpreter.set_tensor(self.input_details[0]['index'], data_row)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                end_time = time.time()

                if "C3" in self.csv_file:
                    prediction = (output_data > 0.5).astype(int).flatten()
                else:
                    prediction = np.argmax(output_data, axis=-1)

            pred_value = prediction[0] if hasattr(prediction, '__getitem__') else prediction
            
            self.tx_process_data(pred_value)

            time_pred = end_time - start_time
            #data = [time_pred, pred_value]

            times[idx] = time_pred

            #self.write_data_file(data)
            idx += 1
        
        print('received records: ', idx)
        print(f'average prediction time: {np.mean(times):.6f}')
    def write_data_file(self, data):
        with open(self.csv_file, "a") as file:
            csv_writer = csv.DictWriter(file, self.fieldnames)
            info = {
                self.fieldnames[0]: data[0],
                self.fieldnames[1]: data[1]
            }
            csv_writer.writerow(info)