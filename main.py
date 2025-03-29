from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from train import train_model

# 获取训练结果
train_losses, test_losses, train_accuracies, test_accuracies, model, test_loader, device = train_model()

timestamp = datetime.now().strftime('%Y%m%d_%H%M')

# 保存训练结果
results = np.array([train_losses, test_losses, train_accuracies, test_accuracies]).T
np.savetxt(f'out/results_{timestamp}.csv', results, delimiter=',', header='Train Loss,Test Loss,Train Accuracy,Test Accuracy', comments='')


## 绘制 loss 和 accuracy 曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')

plt.savefig(f'out/loss_accuracy_curve_{timestamp}.png')
plt.show()


## 绘制混淆矩阵
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)

labels = ['Go', 'Be', 'No', 'Here', 'Hello', 'Thanks', 'Come', 'Eat', 'Walk']
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, ax = plt.subplots(figsize=(10, 10))  # Increase the figure size
cax = ax.matshow(cm_percentage, cmap="Blues")
fig.colorbar(cax, fraction=0.046, pad=0.04)  # Adjust colorbar length
ticks = np.arange(len(labels))
plt.xticks(ticks, labels, rotation=45, fontsize=12, fontname='Arial')
plt.yticks(ticks, labels, fontsize=12, fontname='Arial')
ax.xaxis.set_ticks_position('bottom')  # Move x-axis labels to the bottom
plt.xlabel('Predicted Label', fontsize=14, fontname='Arial')
plt.ylabel('True Label', fontsize=14, fontname='Arial')

# Add percentage sign to the text in the confusion matrix
thresh = cm_percentage.max() / 2  # Threshold for text color
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.annotate(
            f'{cm_percentage[i, j]:.1f}%',
            (j, i),
            ha='center',
            va='center',
            color='white' if cm_percentage[i, j] > thresh else 'black',
            fontsize=12,
            fontname='Arial'
        )

overall_accuracy = np.sum(np.diag(cm)) / np.sum(cm) * 100
plt.title(f'Overall Accuracy: {overall_accuracy:.1f}%', color='blue', weight='bold', fontsize=16, fontname='Arial')
plt.tight_layout()
plt.savefig(f'out/confusion_matrix_{timestamp}.png')
plt.show()