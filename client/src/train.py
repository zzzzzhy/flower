import torch
import numpy as np

# 定义超参数
batch_size = 32 # 批次大小
seq_length = 100 # 序列长度
num_features = 3 # 输入特征的数量
hidden_size = 512 # LSTM的隐藏单元数
output_size = 2 # 输出特征的数量
learning_rate = 0.01 # 学习率
device = torch.device("mps")

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

# 创建模型实例
model = LSTMModel(num_features, hidden_size, output_size)
# 定义损失函数，使用均方误差
criterion = torch.nn.MSELoss()
# 定义优化器，使用随机梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

X = torch.randn(1000, seq_length, num_features)
y = torch.randn(1000, output_size)

# 训练模型，使用10%的数据作为验证集
train_size = int(0.9 * len(X))
val_size = len(X) - train_size
X_train, X_val = torch.utils.data.random_split(X, [train_size, val_size])
y_train, y_val = torch.utils.data.random_split(y, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train.dataset, y_train.dataset), batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val.dataset, y_val.dataset), batch_size=batch_size, shuffle=False)

model.to(device) # 将模型移动到设备上
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
    # 验证阶段
    model.eval()
    val_loss = 0
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
    val_loss = val_loss / len(val_loader.dataset)
    # 打印损失信息
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
