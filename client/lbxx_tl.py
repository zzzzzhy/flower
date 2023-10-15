import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
BATCH_SIZE = 32
_xm = [13.050167224080267, 13.04236343366778, 13.060200668896321, 6.338907469342252, 6.326644370122631, 6.32329988851728, 1016.1259754738015, 1016.1516164994426, 1016.0746934225195, 87.93199554069119, 87.92976588628763, 87.96209587513935, 45.75585284280937, 45.725752508361204, 45.644370122630995, 19.45484949832776, 19.46153846153846, 19.50613154960981,
        6.326644370122631, 6.308807134894091, 6.3076923076923075, 9.285395763656632, 9.267558528428093, 9.274247491638796, 3.117056856187291, 3.1248606465997772, 3.129319955406912, 1019.9253065774805, 1019.9264214046823, 1019.8751393534002, 1012.3110367892976, 1012.3110367892976, 1012.2274247491639, 2.5297324414715705, 2.6036231884057948, 2.5240579710144906]
_xs = [10.93880527996931, 10.965761561911133, 10.950519561488052, 10.528454417888952, 10.55987299759118, 10.565622459773206, 7.650588888188688, 7.628160302814521, 7.58746807102032, 9.29768145347197, 9.40507792650373, 9.387407731845805, 16.01683659069397, 16.100386553626777, 16.132725490076027, 11.53707713084942, 11.550911158031614, 11.522079829966145,
        10.935571588855181, 10.983795727717316, 10.978090927149752, 10.105626030247455, 10.129473203111313, 10.138864985052559, 11.168253553533306, 11.200514381780776, 11.19707911452624, 7.818167358661445, 7.845153003937791, 7.744672395274292, 7.953615581625394, 7.9187780956527325, 7.922957157071328, 8.444558170394911, 8.692138385010162, 8.373725725716474]
xm = np.array(_xm)
xs = np.array(_xs)

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


def tuili(input_data):
    data = (input_data - xm) / xs
    XMin = np.nanmin(data, axis=0)
    XMax = np.nanmax(data, axis=0)
    data = (data - XMin) / (XMax - XMin)
    _data = torch.Tensor(data)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DNN()
    model.load_state_dict(torch.load('src/Regression-best.th'))
    model.eval()
    with torch.no_grad():
        model.to(device)
        output = model(_data)
    return output


if __name__ == "__main__":
    # 参考csv的数据,删掉date和meantempm列就行
    input_data = [-4, -6, -6, -11, -9, -12, 1016, 1022, 1023, 92, 92, 84, 59, 56, 54, 3, 1,
                  2, -13, -12, -13, -4, -6, -6, -16, -13, -18, 1025, 1026, 1025, 1010, 1017, 1019, 0.76, 0, 0]

    tuili(input_data)
