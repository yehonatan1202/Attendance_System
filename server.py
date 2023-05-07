import os
import socket
import cv2
import numpy as np
from utils import load_model, compare_batch, generate_vector
from siamese_network import create_embedding_model, create_distance_model
from db_requests import get_vector, valid_rfid, present

vectors_path = 'vectors'

class Server:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.server_socket = None
        self.rfid = None
        self.rfid_photo = None
        self.model = load_model('siamesemodelv5.h5')
        self.embedding_model = create_embedding_model(self.model)
        self.distance_model = create_distance_model(self.model)

    def start(self):
        # Reset data
        self.rfid = None
        self.rfid_photo = None
        # Create a TCP/IP socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Bind the socket to a specific IP address and port
        self.server_socket.bind((self.ip, self.port))

        while True:
            # Listen for incoming connections
            self.server_socket.listen()
            print(f"Server started on {self.ip}:{self.port}")
            # Wait for a client to connect
            client_socket, client_address = self.server_socket.accept()
            print(f"Client connected: {client_address[0]}:{client_address[1]}")
            for img in os.listdir(vectors_path):
                os.remove(os.path.join(vectors_path, img))
            while True:
                # Recive the massage type: i - image(face), s - string(rfid)
                msg_type = client_socket.recv(1)
                # Handle the data based on its type
                if msg_type == b"i":
                    # Make sure id recived before photo
                    # if not self.rfid_photo:
                    #     print('no id')
                    #     client_socket.send(b"0")
                    #     break
                    
                    # Receive the image size
                    data_size = client_socket.recv(4)
                    size = int.from_bytes(data_size, byteorder='big')

                    # Receive the image data
                    data = b""
                    while len(data) < size:
                        packet = client_socket.recv(size - len(data))
                        if not packet:
                            break
                        data += packet

                    # Convert the byte data to a numpy array
                    img = cv2.imdecode(np.frombuffer(
                        data, dtype=np.uint8), cv2.IMREAD_COLOR)
                   
                    # Save image
                    vectors = [os.path.join(vectors_path,name) for name in os.listdir(vectors_path)]
                    index = len(vectors)
                    cv2.imwrite(f'{vectors_path}/photo_{index}.jpg', img)
                    size = index + 1
                    if(size == 10):
                        match = compare_batch(self.embedding_model, self.distance_model, vectors_path, self.rfid_photo)
                        # Send results and update database
                        if match == False:
                            client_socket.send(b"0")
                        else:
                            present(self.rfid)
                            client_socket.send(b"1")
                        break
                    else:
                        client_socket.send(b"2")

                elif msg_type == b"s":
                    # Receive the string size
                    data_size = client_socket.recv(4)
                    size = int.from_bytes(data_size, byteorder='big')

                    # Receive the string data
                    data = b""
                    while len(data) < size:
                        packet = client_socket.recv(size - len(data))
                        if not packet:
                            break
                        data += packet

                    # Convert the byte data to a string
                    self.rfid = data.decode('utf-8')
                    print(self.rfid)
                    
                    # Send a response to the client
                    if valid_rfid(self.rfid) == True:
                        # Get user photo vector
                        self.rfid_photo = get_vector(self.rfid)
                        client_socket.send(b"1")
                    else:
                        client_socket.send(b"0")
                else:
                    client_socket.close()
                    break

            # Close the client socket
            client_socket.close()


server = Server('192.168.0.249', 6969)
server.start()
