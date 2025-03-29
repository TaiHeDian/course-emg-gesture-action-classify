import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split

from data_processing import load_data, signal_processing
from model import CNNModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model():
    # 加载和处理数据
    data_dict = load_data()
    data = signal_processing(data_dict)

    samples = 235
    # 数据集划分
    X = data.reshape(-1, data.shape[-1])
    y = np.repeat(np.arange(9), samples)  # 生成标签

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 转换为 PyTorch 张量并移动到 device 上
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

    # 初始化模型、损失函数和优化器，并移动模型到 device 上
    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001) #SGD不行
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    num_epochs = 100
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss = running_loss / len(test_loader)
        test_accuracy = correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    return train_losses, test_losses, train_accuracies, test_accuracies, model, test_loader, device

if __name__ == '__main__':
    train_losses, test_losses, train_accuracies, test_accuracies, model, test_loader, device = train_model()
    
    plt.plot(train_accuracies, label='Train Accuracy') 
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.tight_layout()
    plt.show()
