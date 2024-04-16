import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import socket
from datetime import datetime
import pickle

# for debuging
def d_print(str):
    with open("client_debug.txt", 'a') as f:
        f.write(f"{str}\n")

# check if client_id is valid
def client_id_type(client_id):
    valid_client_ids = [f'client{i}' for i in range(1,6)]
    if client_id not in valid_client_ids:
        raise argparse.ArgumentTypeError(f"{client_id} is an invalid clientID. Choose from client1-5.")
    return client_id

# check if port is vlid
def port_client_type(port_client):
    port_client = int(port_client)
    if port_client < 6001 or port_client > 6005:
        raise argparse.ArgumentTypeError("Client port must be between 6001 and 6005.")
    return port_client

# check if opt type is valid
def opt_method_type(opt_method):
    if opt_method not in ['0', '1']:
        raise argparse.ArgumentTypeError("Optimisation must be '0' for Gradient Descent or '1' for Mini-Batch GD.")
    return opt_method


class Client():
    def __init__(self, id,port,opt_method):
        self.id = id
        self.port = port
        self.batch_size = 64
        self.mini_batch_size = 64*5
        self.loss = nn.MSELoss()
        self.iteration = 0
        self.opt_method = opt_method

        self.load_and_preprocess_data()

        self.model = nn.Linear(in_features=8, out_features=1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.receive_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.receive_socket.bind(('127.0.0.1', port))
        self.receive_socket.listen(1)

   
    def logging(self,str):
        with open(f"{self.id}_log.txt", 'a') as f:
            f.write(f"{str}\n")

    def load_data_from_csv(self, csv_path):
       # Read dataset
        df = pd.read_csv(csv_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Normalisation
        feature_ranges = [
            (0, 15),  # MedInc
            (0, 100),  # HouseAge
            (1, 15),  # AveRooms
            (1, 15),  # AveBedrms
            (0, 5000),  # Population
            (1, 10),  # AveOccup
            (32, 42),  # Latitude
            (-124, -114)  # Longitude
        ]

        X_normalized = np.zeros_like(X)
        for i, (min_val, max_val) in enumerate(feature_ranges):
            X_normalized[:, i] = (X[:, i] - min_val) / (max_val - min_val)

        return X_normalized, y

    def train(self, epochs):
        print("Local training...")
        self.model.train()
        
        total_loss = 0  
        total_batches = 0
        if 
        for epoch in range(1, epochs + 1):
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X).squeeze()
                loss = self.loss(output, y)
                total_loss += loss.item()
                total_batches += 1
                loss.backward()
                self.optimizer.step()

        print(f"Training MSE: {total_loss/total_batches}")
        self.logging(f"Training MSE: {total_loss/total_batches}\n")

        d_print("train finished")
        return total_loss/total_batches

    def test(self):
        self.model.eval()
        total_loss = 0
        total_batch = 0
        for x, y in self.testloader:
            y_pred = self.model(x).squeeze()
            total_loss += self.loss(y_pred, y)
            total_batch += 1

        print(f"Testing MSE: {total_loss/total_batch}")
        self.logging(f"Testing MSE: {total_loss/total_batch}")
        d_print("test finished")

        return total_loss/total_batch
    
    def load_and_preprocess_data(self):

        training_path = f'FLData/calhousing_train_{self.id}.csv'
        testing_path = f'FLData/calhousing_test_{self.id}.csv'

        self.X_train, self.y_train = self.load_data_from_csv(training_path)
        self.X_test, self.y_test = self.load_data_from_csv(testing_path)

        # Convert arrays to tensors and create TensorDataset for DataLoader
        self.train_data = TensorDataset(torch.tensor(self.X_train, dtype=torch.float32), torch.tensor(self.y_train, dtype=torch.float32))
        self.test_data = TensorDataset(torch.tensor(self.X_test, dtype=torch.float32), torch.tensor(self.y_test, dtype=torch.float32))

        # Define DataLoader for iterable dataset

        if self.opt_method==0: # GD
            self.trainloader = DataLoader(self.train_data, batch_size = len(self.train_data))
            self.testloader = DataLoader(self.test_data, batch_size = 1)

        else: # mini-batch GD
            self.trainloader = DataLoader(self.train_data, batch_size=self.batch_size)
            self.testloader = DataLoader(self.test_data, batch_size = 1)
        
        d_print(f"(In Client.load_and_preprocess_data) the num of data points: {len(self.train_data)}")
        d_print(f"(In Client.load_and_preprocess_data) the num of batches: {len(self.trainloader)}")

    def hand_shake(self):
        default_timeout = self.socket.gettimeout()
        self.socket.settimeout(20)
        try:
            self.socket.connect(("127.0.0.1", 6000))
        except Exception:
            print("error connecting to the server, terminate!")
            exit(1)


        # Creating the handshake message
        handshake_info = {
            "type": "string",
            "message": f"Handshake: hello, I am {self.id}, length {len(self.train_data)}, port {self.port}"
        }
        # Serializing the message with pickle
        message_sent = pickle.dumps(handshake_info)

        self.socket.send(message_sent)

        d_print(f"(In Client.hand_shake) {self.id} send {message_sent}")

        try:
            response = self.socket.recv(4096)
            response = pickle.loads(response)
            d_print(f"(In Client.hand_shake) the response is {response}")
            if response['type'] ==  'string':
                d_print("(In Client.hand_shake) Success hand-shaking")
                print("successful handshake")
                
            else:
                d_print("(In Client.hand_shake) Failure hand-shaking")
                exit(1)
                
        except socket.timeout:
            print("Error hand-shaking, terminate")
            exit(1)
        finally:
            self.socket.settimeout(default_timeout)
        
        d_print(f"(In Client.hand_shake) The client close client.socket")
        self.socket.close()
        
    def receive_model(self):
        server_socket, server_address = self.receive_socket.accept()
        serialized_model_dict = server_socket.recv(4096)
        model_dict = pickle.loads(serialized_model_dict)

        if len(model_dict) == 1 and model_dict['message'] == "Completed":
            print("Finished Training")
            server_socket.close()
            exit(1)

        print(f"I am {self.id}")
        print("Received new global model")
        d_print(f"(In receive_model) Receive from server: {model_dict}")
        self.model.load_state_dict(model_dict)
        server_socket.close()

    def send_local_model(self):
        print("Sending new local model\n")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(('127.0.0.1', 6000))
        d_print(f"(In send_local_model) socket re-created and re-connected")
 
        message_info = {
        "type": "model",
        "message": f"ClientModel: I am {self.id}",
        "model_param": self.model.state_dict()
        }

        # Serialize the entire dictionary with pickle
        message = pickle.dumps(message_info)
            
        # Send the serialized data
        self.socket.send(message)
        d_print(f"(In send_local_model) Client  sends {message}")
    
if __name__ == "__main__":
    d_print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    parser = argparse.ArgumentParser(description='Federated Learning-Client')
    parser.add_argument('client_id', type=client_id_type, help='Client ID')
    parser.add_argument('client_port', type=port_client_type, help='Port number')
    parser.add_argument('opt_method', type=opt_method_type, help="Optimization method")
    args = parser.parse_args()

    print("Client starting...\n")

    client_id,client_port,opt_method = args.client_id,args.client_port,args.opt_method

    client = Client(client_id,client_port,opt_method)
    client.hand_shake()

    while True:
        client.receive_model()
        client.iteration+=1
        client.logging(f"Iteration{client.iteration}")
        test_loss = client.test()
        train_loss = client.train(10)
        client.send_local_model()
