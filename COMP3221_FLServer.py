import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import argparse


def port_server(id):
    id = int(id)
    if id!=6000:
        raise argparse.ArgumentTypeError(f"Server port must be 6000")
    return id

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
    def __init__(self,sub_sample):
        self.users = []
        self.num_user = 5
        self.model = LinearRegressionModel()
        self.iterations = 100
        self.sub_sample = sub_sample
    

    def send_parameters(server_model, users):

        # send global model to each users
        for user in users:
            pass
          
    def aggregate_parameters(server_model, users, total_train_samples):

        # Clear global model before aggregation
        for param in server_model.parameters():
            param.data = torch.zeros_like(param.data)

        for user in users:
            for server_param, user_param in zip(server_model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * user.train_samples / total_train_samples
        return server_model


    def evaluate(users):
        total_mse = 0
        for user in users:
            total_mse += user.test()
        return total_mse/len(users)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning-Server')

    parser.add_argument('client_port', type=port_server, help='Port number')
    parser.add_argument('sub_client', type=sub_client, help='Sub-client number')
    args = parser.parse_args()

    sub_client_num = args.sub_client
    
    server = Server(sub_client_num)

    # Listening all clients 
    users = []

    # !!! NOT SURE HERE
    total_train_samples = 10000

    for i in range(server.iterations):
        # Send the global model
        server.send_parameters(server.model,users)

        # Recevie the local weights from users
            #....

        # Aggregate the parameters (depends on sub-client)
        server.aggregate_parameters(total_train_samples)








