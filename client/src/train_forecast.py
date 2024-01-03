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
batch_size = 32 # 批次大小
seq_length = 100 # 序列长度
num_features = 3 # 输入特征的数量
hidden_size = 512 # LSTM的隐藏单元数
output_size = 2 # 输出特征的数量
learning_rate = 0.01 # 学习率
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

# 定义模型
class LSTMModel(torch.nn.Module):
    def __init__(self, num_features, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(num_features, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        output = self.linear(lstm_out[:, -1, :])
        return output


def load_data():
    df = pd.read_csv(curdir + '/data.csv').set_index('import_time')
    # X will be a pandas dataframe of all columns except meantempm
    df = pd.read_csv(curdir + '/data.csv',names=["heart_rate", "breath_rate", "sleep_state", "person_id"], sep=",")
# X will be a pandas dataframe of all columns except meantempm
    X = df.groupby('person_id')[['heart_rate', 'breath_rate', 'sleep_state']].apply(lambda x: x.values.tolist()).tolist()
    # split data into training set and a temporary set using sklearn.model_selection.traing_test_split
    X_train_val, X_val= train_test_split(X, test_size=0.1, random_state=23)
    # Standardize X_train_val
    XMean = np.nanmean(X_train_val, axis=0)
    XStd = np.nanstd(X_train_val, axis=0)
    X_train_val = (X_train_val - XMean) / XStd
    XMin = np.nanmin(X_train_val, axis=0)
    XMax = np.nanmax(X_train_val, axis=0)
    X_train_val = (X_train_val - XMin) / (XMax - XMin)
    
    y = torch.randn(2000, output_size)
    X = torch.randn(2000, seq_length, num_features)
    train_size = int(0.9 * len(X))
    val_size = len(X) - train_size
    X_train, X_val = torch.utils.data.random_split(X, [train_size, val_size])
    y_train, y_val = torch.utils.data.random_split(y, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train.dataset, y_train.dataset), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val.dataset, y_val.dataset), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def train(model, device, train_loader, epochs, addr,batchId):
    time_start = time.time()
    # 定义损失函数，使用均方误差
    criterion = torch.nn.MSELoss()
    # 定义优化器，使用随机梯度下降
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    val_MSE = []
    epochs = 100
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            # 将数据移动到设备上，如果有GPU的话
            inputs = inputs.to(device)
            targets = targets.to(device)
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累积损失
            train_loss += loss.item() * inputs.size(0)
        # 计算平均训练损失
        train_loss = train_loss / len(train_loader.dataset)
        val_MSE.append(train_loss)
        if len(val_MSE) == 0 or val_MSE[-1] <= min(np.array(val_MSE)):
                # 如果比之前的mse要小，就保存模型
                torch.save(model.state_dict(), "model.pth".format(val_MSE[-1]))
        # val_plot(val_MSE)
        time_end = time.time()
        print('Training time:', time_end - time_start, 's')
        print('Train Finished')

def test(model, device, val_loader):
    model.eval()
    val_loss = 0
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for inputs, targets in val_loader:
            # 将数据移动到设备上，如果有GPU的话
            inputs = inputs.to(device)
            targets = targets.to(device)
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            # 累积损失
            val_loss += loss.item() * inputs.size(0)
    # 计算平均验证损失
    accuracy = val_loss / len(val_loader.dataset)
    # 打印损失信息
    print('Test Finished ')
    return val_loss, accuracy

