import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import argparse
import socket
import threading
from datetime import datetime

# for debuging
def d_print(str):
    with open("server_debug.txt", 'a') as f:
        f.write(f"{str}\n")

# for checking server port validity
def port_server(id):
    id = int(id)
    if id!=6000:
        raise argparse.ArgumentTypeError(f"Server port must be 6000")
    return id

# for checking sub_client value validity
def sub_client(num):
    num = int(num)
    if num < 0 or num > 4:
        raise argparse.ArgumentTypeError("Sub client number must be within 0-4")
    return num

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.model = nn.Linear(in_features=8, out_features=1)
    
    def forward(self, x):
        return self.model(x)


class Server():
    def __init__(self,sub_sample, server_socket):
        self.clients = {}
        self.max_num_clients = 5
        self.model = LinearRegressionModel()
        self.iterations = 100
        self.sub_sample = sub_sample
        self.socket = server_socket
        self.socket.listen(self.max_num_clients)

    def send_parameters(self, server_model, users):
        # send global model to each users
        for user in users:
            pass
          
    def aggregate_parameters(self, server_model, users, total_train_samples):
        # Clear global model before aggregation
        for param in server_model.parameters():
            param.data = torch.zeros_like(param.data)

        for user in users:
            for server_param, user_param in zip(server_model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * user.train_samples / total_train_samples

        return server_model


    def evaluate(self, users):
        total_mse = 0
        for user in users:
            total_mse += user.test()
        return total_mse/len(users)
    
    # receive messages and redirect to every functions
    def receive_messages(self):
        print("server successfully launched, listening to clients...")
        while True:
            client_socket, client_address = self.socket.accept()
            message = client_socket.recv(2048).decode('utf-8')
            d_print(f"(In Server.receive_messages) {client_address} send {message}")
            self.handshake_reply(message, client_socket)

    # handing handshake, add to self.clients
    def handshake_reply(self, message, client_socket):
        parts = message.split(", ")
        client_id = parts[1].split()[-1]
        client_train_data_size = int(parts[2].split()[-1])
        #d_print(f"(In Server.handshake_reply) The client_id is {client_id}")

        self.clients[client_id] = client_train_data_size
        
        response_message = f"copy, {client_id}"
        d_print(f"(In Server.handshake_reply) Server reply {response_message}")
        client_socket.send(response_message.encode('utf-8'))
        client_socket.close()

if __name__ == "__main__":
    d_print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('server_port', type=port_server, help='Port number for the server')
    parser.add_argument('sub_client', type=sub_client, help='Sub-client number')
    args = parser.parse_args()

    print("Server starting...\n")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', args.server_port))

    server = Server(args.sub_client, server_socket)
    server.receive_messages()










    # # Print weights and biases
    # for name, param in server.model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)


    # # Listening and connect all clients  (wait 30s after first connection)
    # users = []

    # # !!! NOT SURE HERE
    # total_train_samples = 10000

    # for i in range(server.iterations):
    #     # Send the global model
    #     server.send_parameters(server.model,users)

    #     # Recevie the local weights from users


    #     # Aggregate the parameters (depends on sub-client)
    #     server.aggregate_parameters(total_train_samples)

    # # send finish message to all clients
    # # for user in users:
    # #     user.sendall("finish message")






