import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2) # [批量大小, 32, 时间步数]  2 个单位的 padding，确保卷积后的长度不变
        self.pool1 = nn.MaxPool1d(kernel_size=2) # [批量大小, 32, 时间步数 / 2 ]
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2) # [批量大小, 64, 时间步数]
        self.pool2 = nn.MaxPool1d(kernel_size=2) # [批量大小, 64, 时间步数 / 2 ]
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2) #[批量大小, 128, 时间步数]
        
        num = 10000
        for _ in range(2):  # 两个汇聚层
            num = (num + 1) // 2  # 汇聚层缩小时间步数的一半，加1是为了向上取整
        
        self.fc1 = nn.Linear(128 * num, 256)  # 调整全连接层神经元数量 
        self.fc2 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 添加Dropout层
        
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 多维张量展平为一维向量，输入全连接层

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # 创建模型实例
    model = CNNModel()
    print(model)
