import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse

def client_id_type(client_id):
    valid_client_ids = [f'client{i}' for i in range(1,6)]
    if client_id not in valid_client_ids:
        raise argparse.ArgumentTypeError(f"{client_id} is an invalid clientID. Choose from client1-5.")
    return client_id

def port_client_type(port_client):
    port_client = int(port_client)
    if port_client < 6001 or port_client > 6005:
        raise argparse.ArgumentTypeError("Client port must be between 6001 and 6005.")
    return port_client

def opt_method_type(opt_method):
    if opt_method not in ['0', '1']:
        raise argparse.ArgumentTypeError("Optimisation must be '0' for Gradient Descent or '1' for Mini-Batch GD.")
    return opt_method


class Client():
    def __init__(self, id,port, learning_rate, batch_size):
        self.id = id
        self.port = port
        self.batch_size = batch_size
        self.loss = nn.MSELoss()

        self.load_and_preprocess_data()

        # TBC
        self.model = nn.Linear(in_features=8, out_features=1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        
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

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

    def train(self, epochs):
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X).squeeze()
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        return loss.data

    def test(self):
        self.model.eval()
        mse = 0
        for x, y in self.testloader:
            y_pred = self.model(x).squeeze()
           
            mse += self.loss(y_pred, y)

        print(f"MSE of client is {mse}")

        return mse
    
    def load_and_preprocess_data(self):

        training_path = f'FLData/calhousing_train_{self.id}.csv'
        testing_path = f'FLData/calhousing_test_{self.id}.csv'

        self.X_train, self.y_train = self.load_data_from_csv(training_path)
        self.X_test, self.y_test = self.load_data_from_csv(testing_path)

        # Convert arrays to tensors and create TensorDataset for DataLoader
        self.train_data = TensorDataset(torch.tensor(self.X_train, dtype=torch.float32), torch.tensor(self.y_train, dtype=torch.float32))
        self.test_data = TensorDataset(torch.tensor(self.X_test, dtype=torch.float32), torch.tensor(self.y_test, dtype=torch.float32))

        # Define DataLoader for iterable dataset
        self.trainloader = DataLoader(self.train_data, batch_size=self.batch_size)
        self.testloader = DataLoader(self.test_data, batch_size=self.batch_size)

    def updata_local_model(self, model_state):
        self.model.load_state_dict(model_state)
    
    def send_local_model(self):
        pass


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning-Client')
    parser.add_argument('client_id', type=client_id_type, help='Client ID')
    parser.add_argument('client_port', type=port_client_type, help='Port number')
    parser.add_argument('opt_method', type=opt_method_type, help="Optimization method")

    args = parser.parse_args()

    client_id,client_port,opt_method = args.client_id,args.client_port,args.opt_method

    client = Client(client_id,client_port,0.001, 64)

    # local training part
    loss = client.train(10)
    print(round(loss.item(),2))

    # evaulation part

    # assume received model
    model_state = client.model.state_dict()
    client.updata_local_model(model_state)
    print(model_state)
   
   # evaulate 
    client.test()


