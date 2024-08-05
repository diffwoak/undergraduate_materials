from socket import *
serverName = '127.0.0.1'
serverPort = 9999
clientSocket = socket(AF_INET, SOCK_DGRAM)
while True:
    message = input('Input lowercase sentence:')
    if message=='break':break
    clientSocket.sendto(message.encode(), (serverName, serverPort))
    modifiedMessage, serverAddress = clientSocket.recvfrom(2048)
    print(modifiedMessage.decode())
clientSocket.close()