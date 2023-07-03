import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_one(nn.Module):
    def __init__(self, num_class=2):
        super(Net_one, self).__init__()
        self.num_class = num_class
        self.resnet = resnet_block()

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, self.num_class)

    def forward(self, x):
        featureMap, x = self.resnet(x)
        x = self.dropout(x)
        x = self.relu(x)
        out = self.fc2(x)
        return x, out


class Net_one(nn.Module):
    def __init__(self, num_class=2):
        super(Net_one, self).__init__()
        self.num_class = num_class
        self.resnet1 = resnet_block()
        self.resnet2 = resnet_block()

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, self.num_class)

    def forward(self, x1, x2):
        x1 = self.resnet(x1)
        x2 = self.resnet(x2)
        x = torch.cat(x1, x2)
        x = self.dropout(x)
        x = self.relu(x)
        out = self.fc2(x)
        return x, out