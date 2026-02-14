# visualize.py
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_training_data(history, save_dir):
    """保存训练过程数据为 CSV"""
    df = pd.DataFrame(history)
    timestamp = get_timestamp()
    csv_path = os.path.join(save_dir, f"training_log_{timestamp}.csv")
    df.to_csv(csv_path, index_label='epoch')
    print(f"训练数据已保存: {csv_path}")
    return timestamp

def plot_history(history, save_dir, timestamp):
    """绘制训练曲线"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Loss 曲线
    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='#1f77b4', markersize=4)
    ax1.plot(epochs, history['val_loss'], 's--', label='Val Loss', color='#ff7f0e', markersize=4)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Accuracy 曲线
    ax2.plot(epochs, history['train_acc'], 'o-', label='Train Acc', color='#2ca02c', markersize=4)
    ax2.plot(epochs, history['val_acc'], 's--', label='Val Acc', color='#d62728', markersize=4)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    img_path = os.path.join(save_dir, f"learning_curve_{timestamp}.png")
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"曲线图已保存: {img_path}")

def save_confusion_matrix_data(cm, class_names, save_dir, timestamp):
    """保存混淆矩阵数据为 CSV"""
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    csv_path = os.path.join(save_dir, f"confusion_matrix_{timestamp}.csv")
    df.to_csv(csv_path)
    print(f"混淆矩阵数据已保存: {csv_path}")

def plot_confusion_mat(y_true, y_pred, class_names, save_dir, timestamp):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 先保存数据
    save_confusion_matrix_data(cm, class_names, save_dir, timestamp)

    plt.figure(figsize=(10, 8))
    # 使用 Blues 颜色映射，并在格子中显示数值
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                square=True, cbar_kws={"shrink": .8})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    img_path = os.path.join(save_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"混淆矩阵图已保存: {img_path}")
