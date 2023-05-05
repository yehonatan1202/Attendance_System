import socket
import os
import cv2
class Client:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = None
        
    def connect(self):
        # Create a TCP/IP socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect the socket to the server's IP address and port
        self.client_socket.connect((self.server_ip, self.server_port))
        
    def send_photo(self, img):
        # Encode the image as bytes
        img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
        # Send data type
        self.client_socket.send(b"i")
        # Send image size
        size = len(img_bytes)
        self.client_socket.sendall(size.to_bytes(4, byteorder='big'))
        # Send the photo data to the server
        self.client_socket.sendall(img_bytes)
        # Recive the response from the server
        response = self.client_socket.recv(1)
        if response == b"1":
            print('Photo Match')
            return True
        else:
            print('No Photo Match')
            return False

    def send_string(self, message):
        # Encode the string
        data = message.encode('utf-8')
        # Send data type
        self.client_socket.send(b"s")
        # Send string size
        data_size = len(data).to_bytes(4, byteorder='big')
        self.client_socket.send(data_size)
        # Send the string to the server
        self.client_socket.send(data)
        # Recive the response from the server
        response = self.client_socket.recv(1)
        if response == b"1":
            print('ID Match')
            return True
        else:
            print('No ID Match')
            return False
