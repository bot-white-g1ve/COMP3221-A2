import torch
import torch.nn as nn
import argparse 
import socket
import threading
from datetime import datetime
import time
import pickle
import random

# for debuging
def d_print(str):
    pass
    '''
    with open("client_debug.txt", 'a') as f:
        f.write(f"{str}\n")
    '''

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
        self.sub_sample = int(sub_sample)
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('127.0.0.1', port))
        self.socket.listen(self.max_num_clients)
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        self.first_handshake_received = None
        self.current_iteration = 0
        self.num_current_received = 0
    
    # close connections when iterations are fininshed
    def close_server(self):
        final_message = {"message":"Completed"}
        serialized_message = pickle.dumps(final_message)
        
        for client_id, client_info in self.clients.items():
            self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_port = client_info['port']
            try:
                self.send_socket.connect(('127.0.0.1', client_port))
                self.send_socket.send(serialized_message)
            except Exception as e:
                print(f"client {client_id} is down!")
            finally:
                self.send_socket.close()

        self.socket.close()

    # aggregare parameters from clients
    def aggregate_parameters(self):
        if self.iterations == self.current_iteration:
            print("Finished training")
            self.close_server()
            exit(1)

        print("Aggregating new global model")
        # Get the state dictionary of the server model
        server_state_dict = self.model.state_dict()
        d_print(f"(In aggregate_parameters) The original dict_state is {server_state_dict}")
        
        # Initialize an empty state dict to accumulate updates
        aggregated_state_dict = {key: torch.zeros_like(value) for key, value in server_state_dict.items()}

        selected_clients = self.get_joined_users()

        # Determine the number of clients
        if self.sub_sample > 0:
            print(f"sub_sample:{self.sub_sample}, available:{len(selected_clients)}")
            selected_clients = random.sample(selected_clients, min(self.sub_sample, len(selected_clients)))
        
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

         # Update client readiness for the next iteration
        for client_id in self.clients:
            self.clients[client_id]['ready'] = True

    # receive messages and redirect to every functions
    def receive_messages(self):
        print("Server successfully launched, listening to clients...\n")
        while True:
            try:
                if self.first_handshake_received is not None and time.time()-self.first_handshake_received>30:
                    self.socket.settimeout(30.0)  # Set timeout for 30 seconds

                client_socket, client_address = self.socket.accept()
                serialized_data = client_socket.recv(4096)
                if serialized_data:
                    # Deserialize the data using pickle
                    message = pickle.loads(serialized_data)

                else:
                    print("error: empty data")

                d_print(f"(In Server.receive_messages) {client_address} send {message}")

                # receive handshake message
                if message['type'] == 'string' and message['message'].startswith('Handshake: '):
                    current_time = time.time()
                    if self.first_handshake_received is None:
                    # if this is the first client connected
                        self.first_handshake_received = current_time
                        print("The fisrt handshake received, wait for 30 seconds then training begins\n")
                        d_print("(In Server.receive_messages) Waiting for 30 seconds then boardcase")

                        def send_model_after_delay():
                            time.sleep(30)
                            self.send_model_dict()

                        # initiate a thread for the countdown
                        threading.Thread(target=send_model_after_delay).start()
                        
                    self.handshake_reply(message['message'], client_socket)

                # receive client model
                elif message['type'] == 'model' and message['message'].startswith('ClientModel: '):
                    self.clientmodel_handle(message, client_socket)

            except socket.timeout:
                print("\n")
                print("At least one client is down.")
                print("Closing server...")
                self.close_server()
                exit(1)

            except Exception as e:
                print(f"An error occurred: {str(e)}")


    # handing handshake, add to self.clients
    def handshake_reply(self, message, client_socket):
        parts = message.split(", ")
        client_id = parts[1].split()[-1]
        client_train_data_size = int(parts[2].split()[-1])
        client_port = int(parts[3].split()[-1])
        
        # Mark the clients as ready if it joins within registration period
        time_difference = time.time() - self.first_handshake_received
        if time_difference <= 30:
            ready = True
        else:
            print(f"{client_id} will be in the next iteration")
            ready = False

        self.clients[client_id] = {'size': client_train_data_size, 'port': client_port,'ready': ready}
        
        response_message = {
            "type": "string",
            "message": f"copy, {client_id}"
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
    
    # record the model 
    def clientmodel_handle(self, message, client_socket):
        message,model_dict = message['message'],message['model_param']
        parts = message.split(": ")
        client_id = parts[1].split()[-1]
        d_print(f"(In Clientmodel_handle) Client sends model_dict: {model_dict}")
        self.clients[client_id]['model'] = model_dict
        self.clients[client_id]['current_received'] = True
        print(f"Getting local model from {client_id}") 
        self.num_current_received+=1
        if self.num_current_received == len(self.get_joined_users()):
            d_print("(In clientmodel_handle) num_current_received == num_clients, begin aggregate_parameters")
            self.aggregate_parameters()
        client_socket.close()

    # get the ready users
    def get_joined_users(self):
        selected_clients = [client_id for client_id in self.clients if self.clients[client_id]['ready']]
        return selected_clients

if __name__ == "__main__":
    d_print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('server_port', type=port_server, help='Port number for the server')
    parser.add_argument('sub_client', type=sub_client, help='Sub-client number')
    args = parser.parse_args()

    print("Server starting...\n")

    server = Server(args.sub_client, args.server_port)
    server.receive_messages()
