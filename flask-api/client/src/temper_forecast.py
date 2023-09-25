import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import torchvision.transforms as transforms
import time
import pandas as pd
from sklearn.model_selection import train_test_split
# pip install scikit-learn
import matplotlib.pyplot as plt
import torch.utils.data as Data
import json
import requests

from client_config import client_config
from console_utils.console_common import print_receipt_logs_and_txoutput
from client.common import common
from client.common import transaction_common
from client.datatype_parser import DatatypeParser
from console_utils.cmd_account import CmdAccount
from client.signer_impl import Signer_ECDSA
from eth_account.account import Account

contracts_dir = "contracts"
import uuid
def get_mac_address():
    """
    获取本机物理地址，获取本机mac地址
    :return:
    """
    mac=uuid.UUID(int = uuid.getnode()).hex[-12:].upper()
    return "-".join([mac[e:e+2] for e in range(0,11,2)])

# Cred 0xd24180cc0fef2f3e545de4f9aafc09345cd08903 "shareData" "1" "[[-34735, 0, 0, -38811, -24859, 0, 0, 0, -35400, -34118, -30825, 0, -34134, 0, 0, -31372, 0, 0, 0, 0, -38994, -40268, -36570, 0, 0, -36252, -36960, -29528, -36666, 0, 0, 0, -35739, 0, 0, -36435, 0, -32844, 0, 0, -37892, -34644, 0, -5214, 0, 0, 0, -40397, 0, 0, 0, 0, -25964, 0, 0, -34665, -34200, -41271, 0, 0, -30478, 0, 0, -42792, 0, -39536, -25307, 0, -37606, 0, -33774, 0, 0, -38091, 0, 0, -3950, 0, 0, 0, 0, 0, 0, 0, -36714, 0, 0, -32335, 0, -15880, 0, 0, 0, 0, 0, 0, -5972, -37087, 0, 0, -39449, 0, 24, 0, -23541, 0, 0, -37282, -31862, -35911, 0, 0, 0, -33218, 0, 0, 0, 0, 0, 0, -38225, 0, 0, -28504, 0, 0, 0, 0]]"
def upload_data(contractname, address , fn_name, fn_args):
    key_file = "{}/{}".format(client_config.account_keyfile_path, client_config.account_keyfile)
    if not os.access(key_file, os.F_OK):
        CmdAccount.create_ecdsa_account(get_mac_address(),'123456')
        
    with open(key_file, "r") as dump_f:
        keytext = json.load(dump_f)
        privkey = Account.decrypt(keytext, '123456')
        ac2 = Account.from_key(privkey)
        res = requests.post(client_config.node+'/register',data={'address':ac2.address})
        print("register:\t", res.text)
        
    tx_client = transaction_common.TransactionCommon(
        address, contracts_dir, contractname
    )
    # print("INFO>> client info: {}".format(tx_client.getinfo()))
    # print(
    #     "INFO >> sendtx {} , address: {}, func: {}, args:{}".format(
    #         contractname, address, fn_name, fn_args
    #     )
    # )
    try:
        # from_account_signer = None
        from_account_signer = Signer_ECDSA.from_key_file(
           key_file, "123456")
        # print(keypair.address)
        # 不指定from账户，如需指定，参考上面的加载，或者创建一个新的account，
        # 参见国密（client.GM_Account）和非国密的account管理类LocalAccount
        (receipt, output) = tx_client.send_transaction_getReceipt(
            fn_name, fn_args, from_account_signer=from_account_signer)
        data_parser = DatatypeParser(tx_client.contract_abi_path)
        # 解析receipt里的log 和 相关的tx ,output
        print_receipt_logs_and_txoutput(tx_client, receipt, "", data_parser)
    except Exception as e:
        common.print_error_msg("sendtx", e)
        
EPOCH = 100  # train the training data n times, to save time, we just train 1 epoch
LR = 0.001  # learning rate
BATCH_SIZE = 10
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super(MyDataset, self).__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index):
        X, y = torch.Tensor(self.X[index]), torch.Tensor([self.y[index]])
        return X, y

    def __len__(self):
        return len(self.X)

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(36, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, 1)

    def forward(self, x):
        y = self.layer1(x)
        y = F.relu(y)
        y = self.layer2(y)
        y = F.relu(y)
        y = self.layer3(y)
        y = F.relu(y)
        y = self.layer4(y)
        y = F.relu(y)
        y = self.layer5(y)
        return y

def val_plot(total_loss):
    x = range(len(total_loss))
    plt.plot(x, total_loss, label='Val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Val_loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Val_loss.png')

def load_data():
    df = pd.read_csv('weather_data.csv').set_index('date')
    # X will be a pandas dataframe of all columns except meantempm
    X = df[[col for col in df.columns if col != 'meantempm']].values
    # y will be a pandas series of the meantempm
    y = df['meantempm'].values.astype(np.float64)
    # split data into training set and a temporary set using sklearn.model_selection.traing_test_split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=23)
    # Standardize X_train_val
    XMean = np.nanmean(X_train_val, axis=0)
    XStd = np.nanstd(X_train_val, axis=0)
    X_train_val = (X_train_val - XMean) / XStd
    XMin = np.nanmin(X_train_val, axis=0)
    XMax = np.nanmax(X_train_val, axis=0)
    X_train_val = (X_train_val - XMin) / (XMax - XMin)
    # split train_val into training set and val set using sklearn.model_selection.traing_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=1 / 9, random_state=23)
    X_test = (X_test - XMean) / XStd
    XMin = np.nanmin(X_test, axis=0)
    XMax = np.nanmax(X_test, axis=0)
    X_test = (X_test - XMin) / (XMax - XMin)

    # print("Training instances   {}, Training features   {}".format(X_train.shape[0], X_train.shape[1]))
    # print("Validation instances {}, Validation features {}".format(X_val.shape[0], X_val.shape[1]))
    # print("Testing instances    {}, Testing features    {}".format(X_test.shape[0], X_test.shape[1]))

    train_data = MyDataset(X_train, y_train)
    val_data = MyDataset(X_val, y_val)
    test_data = MyDataset(X_test, y_test)

    train_loader = Data.DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader

def train(model, device, train_loader, val_loader, epochs,addr):
    time_start = time.time()
    # model = DNN()
    # print(model)  # net architecture
    # Loss and optimizer
    criterion = nn.MSELoss(reduction='mean')

    # model.to(device)
    criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # LR=0.001
    val_MSE = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for step, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if step % 10 == 9:
                # print('[%d,%5d] loss: %.3f' % (epoch + 1, (step + 1)*10, train_loss / 100))
                # Batch size=10,所以每训练100个数据输出一次loss
                train_loss = 0.0
        model.eval()
        val_loss = 0.
        with torch.no_grad():  # 不需要更新模型，不需要梯度
            for step, (data, label) in enumerate(val_loader):
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                loss = criterion(output, label)
                val_loss += loss.item()
            val_MSE.append(val_loss / val_loader.dataset.X.shape[0])
        model.train()
        if len(val_MSE) == 0 or val_MSE[-1] <= min(np.array(val_MSE)):
            for name, param in model.named_parameters():
                if param.grad is not None and name == "layer5.weight":
                    if addr:
                        upload_data('Cred',addr,'shareData',param.grad.tolist())
                    else:
                        print('addr is None')
                    print(f"{name}: {param.grad}")
            # 如果比之前的mse要小，就保存模型
            # print("Best model on epoch: {}, val_mse: {:.4f}".format(epoch, val_MSE[-1]))
            torch.save(model.state_dict(), "Regression-best-{:.4f}.th".format(val_MSE[-1]))
        # val_plot(val_MSE)
        time_end = time.time()
        print('Training time:', time_end - time_start, 's')
        print('Train Finished')

def test(model, device, test_loader):
    # model = DNN()
    # model.load_state_dict(torch.load('Simple-DNN-on-weather-forcast/Regression-best.th'))
    # model.to(device)
    criterion = nn.MSELoss(reduction='mean')
    with torch.no_grad():
        test_loss, test_step = 0, 0
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            test_loss += loss.item()
            correct += (torch.max(output.data, 1)[1] == label).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print('Test Finished ')
    return loss, accuracy
        # print("Mse of the best model on the test data is: {:.4f}".format(test_loss / X_test.shape[0]))
# if __name__ == "__main__":
#     EPOCH = 100  # train the training data n times, to save time, we just train 1 epoch
#     LR = 0.001  # learning rate
#     BATCH_SIZE=10
