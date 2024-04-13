import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import argparse
import socket
import threading
from datetime import datetime
import pickle
import time

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
    def __init__(self,sub_sample, port):
        self.clients = {}
        self.max_num_clients = 5
        self.model = LinearRegressionModel()
        self.iterations = 100
        self.sub_sample = sub_sample
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('127.0.0.1', port))
        self.socket.listen(self.max_num_clients)
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        self.first_handshake_received = False
        self.current_iterations = 0
          
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
        print("Server successfully launched, listening to clients...\n")
        while True:
            client_socket, client_address = self.socket.accept()
            message = client_socket.recv(2048).decode('utf-8')
            d_print(f"(In Server.receive_messages) {client_address} send {message}")
            if message.startswith('Handshake: '):
                if not self.first_handshake_received:
                # if this is the first client connected
                    self.first_handshake_received = True
                    print("The fisrt handshake received, wait for 30 seconds then training begins")
                    def send_model_after_delay():
                        time.sleep(30)
                        self.send_model_dict()
                    threading.Thread(target=send_model_after_delay).start()
                self.handshake_reply(message, client_socket)

    # handing handshake, add to self.clients
    def handshake_reply(self, message, client_socket):
        parts = message.split(", ")
        client_id = parts[1].split()[-1]
        client_train_data_size = int(parts[2].split()[-1])
        client_port = int(parts[3].split()[-1])
        #d_print(f"(In Server.handshake_reply) The client_id is {client_id}")

        self.clients[client_id] = {'size': client_train_data_size, 'port': client_port}
        
        response_message = f"copy, {client_id}"
        d_print(f"(In Server.handshake_reply) Server reply {response_message}")
        client_socket.send(response_message.encode('utf-8'))
        client_socket.close()
    
    # send the combined model to all clients
    def send_model_dict(self):
        print("Broadcasting new global model\n")
        model_dict = self.model.state_dict()
        serialized_model_dict = pickle.dumps(model_dict)
        for client in self.clients.keys():
            client_port = self.clients[client]['port']
            self.send_socket.connect(('127.0.0.1', client_port))
            message = f"ServerSend: {serialized_model_dict}"
            self.send_socket.send(message.encode('utf-8'))
            d_print(f"(In send_model_dict) Server sends: {message}")
            d_print(f"(In send_model_dict) The message size is: {len(message.encode('utf-8'))}")

if __name__ == "__main__":
    d_print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('server_port', type=port_server, help='Port number for the server')
    parser.add_argument('sub_client', type=sub_client, help='Sub-client number')
    args = parser.parse_args()

    print("Server starting...\n")

    server = Server(args.sub_client, args.server_port)
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






