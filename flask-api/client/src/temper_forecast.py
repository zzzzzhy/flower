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
from bcos3sdk.bcos3client import Bcos3Client
from bcos3sdk.transaction_status import TransactionStatus
from client.common import common
from client.datatype_parser import DatatypeParser
from client.signer_impl import Signer_ECDSA
curdir = os.path.dirname(os.path.abspath(__file__))

tx_client = None
def init():
    global tx_client
    key_file = "{}/{}".format(client_config.account_keyfile_path,
                              client_config.account_keyfile)
    Bcos3Client.default_from_account_signer = Signer_ECDSA.from_key_file(
        key_file, client_config.account_password)
    tx_client = Bcos3Client()

# Cred 0xd24180cc0fef2f3e545de4f9aafc09345cd08903 "shareData" "1" "[[-34735, 0, 0, -38811, -24859, 0, 0, 0, -35400, -34118, -30825, 0, -34134, 0, 0, -31372, 0, 0, 0, 0, -38994, -40268, -36570, 0, 0, -36252, -36960, -29528, -36666, 0, 0, 0, -35739, 0, 0, -36435, 0, -32844, 0, 0, -37892, -34644, 0, -5214, 0, 0, 0, -40397, 0, 0, 0, 0, -25964, 0, 0, -34665, -34200, -41271, 0, 0, -30478, 0, 0, -42792, 0, -39536, -25307, 0, -37606, 0, -33774, 0, 0, -38091, 0, 0, -3950, 0, 0, 0, 0, 0, 0, 0, -36714, 0, 0, -32335, 0, -15880, 0, 0, 0, 0, 0, 0, -5972, -37087, 0, 0, -39449, 0, 24, 0, -23541, 0, 0, -37282, -31862, -35911, 0, 0, 0, -33218, 0, 0, 0, 0, 0, 0, -38225, 0, 0, -28504, 0, 0, 0, 0]]"
def upload_data(contractname, address , fn_name, fn_args):
    try:
        abiparser = DatatypeParser(
            f"{tx_client.config.contract_dir}/{contractname}.abi")
        # (contract_abi,args) = abiparser.format_abi_args(fn_name,fn_args)
        args = fn_args
        #print("sendtx:",args)
        result = tx_client.sendRawTransaction(
            address, abiparser.contract_abi, fn_name, args)
        # 解析receipt里的log 和 相关的tx ,output
        print(f"Transaction result >> \n{result}")
        status = result['status']
        print(f"Transaction Status >> {status}")
        if not TransactionStatus.isOK(status):
            print("! transaction ERROR", TransactionStatus.get_error_message(status))
        output = result['output']
        output = abiparser.parse_output(fn_name, output)
        print(f"Transaction Output >> {output}")
        # if "logEntries" in result:
        #     logs = abiparser.parse_event_logs(result["logEntries"])
        #     print("transaction receipt events >>")
        #     n = 1
        #     for log in logs:
        #         print(f"{n} ):{log['eventname']} -> {log['eventdata']}")
        #         n = n + 1
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


def load_data():
    df = pd.read_csv(curdir + '/weather_data.csv').set_index('date')
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

def train(model, device, train_loader, val_loader, epochs, addr,batchId):
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
                        grad = np.trunc(param.grad * 10000)
                        # print(f"{name}: {grad.numpy().astype(dtype=int).tolist()}")
                        upload_data('Cred', addr, 'shareData', [
                                    grad.numpy().astype(dtype=int).tolist(),batchId])
                    else:
                        print('addr is None')
                    #print(f"{name}: {param.grad}")
            # 如果比之前的mse要小，就保存模型
            # print("Best model on epoch: {}, val_mse: {:.4f}".format(epoch, val_MSE[-1]))
            torch.save(model.state_dict(), curdir
                       + "/Regression-best-{:.4f}.th".format(val_MSE[-1]))
        # val_plot(val_MSE)
        time_end = time.time()
        print('Training time:', time_end - time_start, 's')
        print('Train Finished')

def test(model, device, test_loader):
    # model = DNN()
    # model.load_state_dict(torch.load('Simple-DNN-on-weather-forcast/Regression-best.th'))
    # model.to(device)
    correct = 0
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
    return test_loss, accuracy
    # print("Mse of the best model on the test data is: {:.4f}".format(test_loss / X_test.shape[0]))
# if __name__ == "__main__":
#     EPOCH = 100  # train the training data n times, to save time, we just train 1 epoch
#     LR = 0.001  # learning rate
#     BATCH_SIZE=10
