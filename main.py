import torch
from data_preprocessing import prepare_data
from model              import CNN1D
from train_eval         import train_and_evaluate
from visualize          import plot_metrics, plot_confusion


# --------- 用户配置 ---------
data_files = {
    "class_1": r"data\1.txt",
    "class_2": r"data\2.txt",
    "class_3": r"data\3.txt",
    "class_4": r"data\4.txt",
    "class_5": r"data\5.txt",
}
signal_length = 2500
test_size     = 0.2
random_state  = 42
batch_size    = 32
num_epochs    = 200
learning_rate = 1e-4
# ----------------------------

# 1. 数据预处理
train_loader, test_loader, encoder, n_classes = prepare_data(
    data_files, signal_length,
    test_size=test_size,
    random_state=random_state,
    batch_size=batch_size
)

# 2. 构建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = CNN1D(signal_length=signal_length, n_classes=n_classes)

# 3. 训练与评估
train_losses, test_accuracies, all_preds, all_labels = train_and_evaluate(
    model, train_loader, test_loader,
    device,
    num_epochs=num_epochs,
    lr=learning_rate
)

# 4. 可视化
plot_metrics(train_losses, test_accuracies)
plot_confusion(all_labels, all_preds, encoder.categories_[0])
