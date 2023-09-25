import warnings
from collections import OrderedDict
import os
import json
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, ToTensor
from src.temper_forecast import train, load_data, test, DNN
import requests

from eth_account.account import Account
from console_utils.cmd_account import CmdAccount
from client_config import client_config

curdir = os.path.dirname(os.path.abspath(__file__))
contracts_dir = "contracts"
import uuid
def get_mac_address():
    """
    获取本机物理地址，获取本机mac地址
    :return:
    """
    mac=uuid.UUID(int = uuid.getnode()).hex[-12:].upper()
    # return 'pyaccount'
    return "-".join([mac[e:e+2] for e in range(0,11,2)])

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################


warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = DNN().to(DEVICE)
train_loader, val_loader, test_loader = load_data()


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, DEVICE, train_loader, val_loader, epochs=100,addr=config.get('address',None))
        return self.get_parameters(config={}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, DEVICE, test_loader)
        return loss, len(test_loader.dataset), {"accuracy": accuracy}


if __name__ == "__main__":
    # Start Flower client
    key_file = "{}/{}".format(client_config.account_keyfile_path, client_config.account_keyfile)
    if not os.access(key_file, os.F_OK):
        CmdAccount.create_ecdsa_account(get_mac_address(),client_config.account_password)
        
    with open(key_file, "r") as dump_f:
        keytext = json.load(dump_f)
        privkey = Account.decrypt(keytext, client_config.account_password)
        ac2 = Account.from_key(privkey)
        res = requests.post(client_config.node+'/register',json={'address':ac2.address})
        print("register:\t", res.text)
        
    fl.client.start_numpy_client(
        server_address="flwr-server:8080",
        client=FlowerClient(),
    )
    # print('Starting')
