import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import argparse
import socket
import threading
from datetime import datetime
import time
import pickle
import random

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

class Server():
    def __init__(self,sub_sample, port):
        self.clients = {}
        self.max_num_clients = 5
        self.model = nn.Linear(in_features=8, out_features=1)
        self.iterations = 100
        self.sub_sample = sub_sample
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('127.0.0.1', port))
        self.socket.listen(self.max_num_clients)
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        self.first_handshake_received = False
        self.current_iteration = 0
        self.num_current_received = 0
          
    def aggregate_parameters(self):
        print("Aggregating new global model")
        # Get the state dictionary of the server model
        server_state_dict = self.model.state_dict()
        d_print(f"(In aggregate_parameters) The original dict_state is {server_state_dict}")
        # Initialize an empty state dict to accumulate updates
        aggregated_state_dict = {key: torch.zeros_like(value) for key, value in server_state_dict.items()}
        # Determine the number of clients to subsample or include all
        if self.sub_sample > 0:
            selected_clients = random.sample(list(self.clients.keys()),self.sub_sample)
        else:
            selected_clients = list(self.clients.keys())
        # Calculate the total training samples for the selected clients
        total_samples_for_aggregation = sum(self.clients[client_id]['size'] for client_id in selected_clients)
        # Accumulate updates only from the selected clients
        for client_id in selected_clients:
            client = self.clients[client_id]
            user_state_dict = client['model']
            for key in server_state_dict:
                aggregated_state_dict[key] += user_state_dict[key] * client['size'] / total_samples_for_aggregation
        # Load the aggregated state dict back into the server model
        d_print(f"(In aggregate_parameters) The aggregated_state_dict is {aggregated_state_dict}")
        self.model.load_state_dict(aggregated_state_dict, strict=True)
        self.send_model_dict()

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
            serialized_data = client_socket.recv(4096)
            if serialized_data:
                # Deserialize the data using pickle
                message = pickle.loads(serialized_data)

            else:
                print("error: empty data")

            d_print(f"(In Server.receive_messages) {client_address} send {message}")

            if message['type'] == 'string' and message['sentence'].startswith('Handshake: '):
                if not self.first_handshake_received:
                # if this is the first client connected
                    self.first_handshake_received = True
                    print("The fisrt handshake received, wait for 30 seconds then training begins\n")
                    d_print("(In Server.receive_messages) Waiting for 30 seconds then boardcase")
                    def send_model_after_delay():
                        time.sleep(5) #deb
                        self.send_model_dict()
                    threading.Thread(target=send_model_after_delay).start()
                    
                self.handshake_reply(message['sentence'], client_socket)

            elif message['type'] == 'model' and message['sentence'].startswith('ClientModel: '):
                self.clientmodel_handle(message, client_socket)

    # handing handshake, add to self.clients
    def handshake_reply(self, message, client_socket):
        parts = message.split(", ")
        client_id = parts[1].split()[-1]
        client_train_data_size = int(parts[2].split()[-1])
        client_port = int(parts[3].split()[-1])

        self.clients[client_id] = {'size': client_train_data_size, 'port': client_port}
        
        response_message = {
            "type": "string",
            "sentence": f"copy, {client_id}"
            }
        # Serializing the message with pickle
        response_message = pickle.dumps(response_message)
        
        d_print(f"(In Server.handshake_reply) Server reply {response_message}")
        client_socket.send(response_message)
        client_socket.close()
    
    # send the combined model to all clients
    def send_model_dict(self):
        print("Broadcasting new global model\n")
        model_dict = self.model.state_dict()
        serialized_model_dict = pickle.dumps(model_dict)
        for client in self.clients.keys():
            self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_port = self.clients[client]['port']
            self.send_socket.connect(('127.0.0.1', client_port))
            self.send_socket.send(serialized_model_dict)
            self.clients[client]['current_received'] = False # Haven't received the client's model dict in this iteration
            d_print(f"(In send_model_dict) Server sends: {model_dict}")
            d_print(f"(In send_model_dict) The message size is: {len(serialized_model_dict)}")
            self.send_socket.close()

        self.current_iteration+=1
        self.num_current_received = 0
        d_print(f"(In send_model_dict) Start iteration {self.current_iteration}")
        print(f"Global Iteration: {self.current_iteration}")
        print(f"Total Number of Clients: {len(self.clients.keys())}")
    
    def clientmodel_handle(self, message, client_socket):
        
        message,model_dict = message['sentence'],message['model_param']
        parts = message.split(": ")
        client_id = parts[1].split()[-1]
        d_print(f"(In Clientmodel_handle) Client sends model_dict: {model_dict}")
        self.clients[client_id]['model'] = model_dict
        self.clients[client_id]['current_received'] = True
        print(f"Getting local model from {client_id}")
        self.num_current_received+=1
        if self.num_current_received == len(self.clients.keys()):
            d_print("(In clientmodel_handle) num_current_received == num_clients, begin aggregate_parameters")
            self.aggregate_parameters()
        client_socket.close()

if __name__ == "__main__":
    d_print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('server_port', type=port_server, help='Port number for the server')
    parser.add_argument('sub_client', type=sub_client, help='Sub-client number')
    args = parser.parse_args()

    print("Server starting...\n")

    server = Server(args.sub_client, args.server_port)
    server.receive_messages()
