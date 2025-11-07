# model.py

import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, signal_length: int, n_classes: int):
        super().__init__()
        # 卷积层 + BN + 汇聚
        self.conv1 = nn.Conv1d(1,  32, kernel_size=5,  padding=2)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, padding=5)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.AvgPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(2)

        # 动态计算展平后特征数
        L = signal_length
        for _ in range(4):
            # 卷积后长度 = L + 2*padding - kernel_size + 1
            k = [5,11,3,3][_]
            p = [2,5,1,1][_]
            L = (L + 2*p - k) + 1
            L //= 2  # 汇聚一律 stride=2

        flat_feats = 256 * L

        # 全连接层
        self.fc1 = nn.Linear(flat_feats, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
