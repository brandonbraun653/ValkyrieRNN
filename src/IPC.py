import sys
import socket


class TCPSocket:
    def __init__(self, host_ip, port_num):
        self.host = host_ip
        self.port = port_num
        self.tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcpServer.settimeout(10)

        self.connection = None
        self.client_addr = None

    def connect(self):
        self.tcpServer.bind((self.host, self.port))
        self.tcpServer.listen(1)

        print('Waiting for connection...')
        try:
            self.connection, self.client_addr = self.tcpServer.accept()
            print('Connected by client', self.client_addr)

        except socket.timeout:
            print("Did not receive connection in time. Cannot perform inference.")
            sys.exit(1)

    def close(self):
        self.connection.close()
        print('Connection closed by server')

    def receive(self, len):
        return str(self.connection.recv(len))

    def send(self, data_str):
        self.connection.send(data_str.encode('utf-8'))
