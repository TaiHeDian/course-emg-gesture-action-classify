# main.py
import json
import random
from functools import partial

import numpy as np
import optuna
import torch.nn as nn
import torch.optim as optim

from config import *
from dataset import DataProcessor
from models import OptimizedCNN
from trainer import evaluate, train_epoch
from visualize import plot_confusion_mat, plot_history, save_training_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证 cudnn 结果一致
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"已固定全局随机种子: {seed}")

def save_params(params, filepath):
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"最佳参数已保存至: {filepath}")

def load_params(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def objective(trial, data_processor):
    params = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'n_filters': trial.suggest_categorical('n_filters', [16, 32, 64]),
        'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7, 11]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5)
    }
    
    train_loader, val_loader = data_processor.get_loaders(batch_size=params['batch_size'])
    
    model = OptimizedCNN(
        n_classes=data_processor.n_classes,
        n_filters=params['n_filters'],
        kernel_size=params['kernel_size'],
        dropout=params['dropout']
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(NUM_EPOCHS_OPTUNA):
        train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_acc

def main():
    # 1. 初始化种子
    set_seed(RANDOM_STATE)
    
    processor = DataProcessor(DATA_FILES, SIGNAL_LENGTH, TEST_SIZE, RANDOM_STATE, skip_rows=SKIP_ROWS)
    best_params = None

    # 2. 检查是否有已保存的参数
    saved_params = load_params(PARAMS_FILE)
    if saved_params:
        print(f"\n检测到已保存的最佳参数: {PARAMS_FILE}")
        print(f"参数内容: {saved_params}")
        choice = input("是否直接使用该参数？(y/n) [默认为 y]: ").strip().lower()
        if choice in ['', 'y', 'yes']:
            best_params = saved_params
        else:
            print("用户选择重新搜索参数...")

    # 3. 如果没有参数或用户选择重跑，则运行 Optuna
    if best_params is None:
        print("\n--- 开始 Optuna 超参数搜索 ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(partial(objective, data_processor=processor), n_trials=N_TRIALS)
        best_params = study.best_params
        save_params(best_params, PARAMS_FILE)
    
    print(f"\n--- 使用参数进行最终训练 (Epochs: {NUM_EPOCHS_FINAL}) ---")
    print(f"当前参数: {best_params}")
    
    # 再次固定种子，确保最终训练过程也是可复现的
    set_seed(RANDOM_STATE)

    train_loader, test_loader = processor.get_loaders(batch_size=best_params['batch_size'])
    
    model = OptimizedCNN(
        n_classes=processor.n_classes,
        n_filters=best_params['n_filters'],
        kernel_size=best_params['kernel_size'],
        dropout=best_params['dropout']
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, NUM_EPOCHS_FINAL + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
        
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{NUM_EPOCHS_FINAL} | "
                  f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    print("\n--- 结果保存与可视化 ---")
    # 保存数据并获取时间戳
    timestamp = save_training_data(history, RESULTS_DIR)
    
    # 绘图
    plot_history(history, RESULTS_DIR, timestamp)
    
    class_names = list(DATA_FILES.keys())
    plot_confusion_mat(labels, preds, class_names, RESULTS_DIR, timestamp)
    
    print(f"所有结果已保存至 {RESULTS_DIR} 文件夹")

if __name__ == "__main__":
    main()
