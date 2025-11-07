# train_eval.py

import torch
import torch.nn as nn

def train_and_evaluate(model: nn.Module,
                       train_loader,
                       test_loader,
                       device,
                       num_epochs: int = 200,
                       lr: float = 1e-4):
    """
    训练并评估模型，返回：
      - train_losses: 每 epoch 的训练损失列表
      - test_accuracies: 每 epoch 的测试准确率列表
      - all_preds: 测试集上所有预测标签
      - all_labels: 测试集上所有真实标签
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, test_accuracies = [], []

    for epoch in range(1, num_epochs+1):
        # —— 训练 —— #
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(Xb)
            loss = criterion(out, torch.argmax(yb, dim=1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        # —— 测试 —— #
        model.eval()
        correct, count = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out   = model(Xb)
                pred  = torch.argmax(out, dim=1)
                true  = torch.argmax(yb, dim=1)
                correct += (pred == true).sum().item()
                count   += pred.size(0)
                all_preds .extend(pred.cpu().numpy())
                all_labels.extend(true.cpu().numpy())

        test_acc = correct / count
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch}/{num_epochs}  "
              f"Train Loss: {train_losses[-1]:.4f}  "
              f"Test Acc: {test_acc:.4f}")

    return train_losses, test_accuracies, all_preds, all_labels
