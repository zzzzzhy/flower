import torch
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
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
# 定义超参数
batch_size = 32 # 批次大小
seq_length = 100 # 序列长度
num_features = 3 # 输入特征的数量
hidden_size = 512 # LSTM的隐藏单元数
output_size = 2 # 输出特征的数量
learning_rate = 0.01 # 学习率
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建模型实例
model = LSTMModel(num_features, hidden_size, output_size)
model.load_state_dict(torch.load('client/src/model.pth'))
model.eval()
data = torch.Tensor([np.array([[34.164078,11.139115,1],
[57.633438,13.557162,2],
[58.245022,15.816837,1],
[43.49895,11.303635,3]]).tolist()])
with torch.no_grad():
    model.to(device)
    output = model(data)
print(output*100)
