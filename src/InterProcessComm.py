import sys
import socket


class TCPSocket:
    def __init__(self, host_ip, port_num):
        self._host = host_ip
        self._port = port_num
        self._tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._tcpServer.settimeout(10)

        self._connection = None
        self._client_addr = None

    def connect(self):
        self._tcpServer.bind((self._host, self._port))
        self._tcpServer.listen(1)

        print('Waiting for connection...')
        try:
            self._connection, self._client_addr = self._tcpServer.accept()
            print('Connected by client', self._client_addr)
            return True

        except socket.timeout:
            print("Did not receive connection in time. Cannot perform inference.")
            sys.exit(1)

    def close(self):
        self._connection.close()
        print('Connection closed by server')

    def receive(self, len):
        return str(self._connection.recv(len))

    def send(self, data_str):
        self._connection.send(data_str.encode('utf-8'))

if __name__ == "__main__":
    import json
    from pprint import pprint

    data = json.load(open('C:\\Users\\Valkyrie\\Desktop\\TEMP\\data.json'))
    pprint(data)

    print(data["timeRise"])
    print(data['pi'])