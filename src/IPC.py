import socket


class TCPSocket:
    def __init__(self, host_ip, port_num):
        self.host = host_ip
        self.port = port_num
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.connection = None
        self.client_addr = None

    def connect(self):
        self.socket.bind((self.host, self.port))
        self.listen(1)

        print('Waiting for connection...')
        self.connection, self.client_addr = self.socket.accept()
        print('Connected by client', self.client_addr)

    def close(self):
        self.connection.close()
        print('Connection closed by server')

    def receive(self, len):
        return str(self.connection.recv(len))

    def send(self, data_str):
        self.connection.send(data_str.encode('utf-8'))
