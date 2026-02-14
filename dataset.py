# dataset.py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset


class DataProcessor:
    def __init__(self, data_files, signal_length=2500, test_size=0.2, random_state=42, skip_rows=3):
        self.data_files = data_files
        self.signal_length = signal_length
        self.test_size = test_size
        self.random_state = random_state
        self.skip_rows = skip_rows
        self.encoder = OneHotEncoder(sparse_output=False)
        self.n_classes = len(data_files)

    def _load_and_segment(self, path):
        """读取并按逻辑切分/补零数据"""
        try:
            # 修改点：增加 skiprows 跳过表头，去除 delimiter=" " 以支持不规则空格
            samples = np.loadtxt(path, skiprows=self.skip_rows)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros((0, self.signal_length))

        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        # 核心切分逻辑
        if samples.shape[1] != self.signal_length:
            segs = []
            for row in samples:
                L = row.shape[0]
                n_seg = max(1, (L + self.signal_length - 1) // self.signal_length)
                for i in range(n_seg):
                    start = i * self.signal_length
                    end = start + self.signal_length
                    seg = row[start:end]
                    if seg.shape[0] < self.signal_length:
                        pad_width = self.signal_length - seg.shape[0]
                        seg = np.pad(seg, (0, pad_width), mode='constant', constant_values=0)
                    segs.append(seg)
            return np.vstack(segs) if segs else np.zeros((0, self.signal_length))
        return samples

    def get_loaders(self, batch_size=32):
        X_tr_list, y_tr_list = [], []
        X_te_list, y_te_list = [], []

        for class_idx, (label, path) in enumerate(self.data_files.items()):
            samples = self._load_and_segment(path)
            if samples.shape[0] == 0: continue

            X_tr, X_te = train_test_split(
                samples, test_size=self.test_size,
                random_state=self.random_state, shuffle=True
            )
            
            X_tr_list.append(X_tr)
            X_te_list.append(X_te)
            y_tr_list.append(np.full(X_tr.shape[0], class_idx))
            y_te_list.append(np.full(X_te.shape[0], class_idx))

        # 再次检查是否有数据，避免 vstack 报错
        if not X_tr_list:
            raise ValueError("没有加载到任何有效数据！请检查 config.py 中的路径或 dataset.py 中的 skip_rows 设置。")

        X_train = np.vstack(X_tr_list)
        X_test = np.vstack(X_te_list)
        y_train = np.concatenate(y_tr_list)
        y_test = np.concatenate(y_te_list)

        # One-Hot Encoding
        y_train_enc = self.encoder.fit_transform(y_train.reshape(-1, 1))
        y_test_enc = self.encoder.transform(y_test.reshape(-1, 1))

        # 转 Tensor & 标准化
        X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) # (N, 1, L)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

        mean, std = X_train_t.mean(), X_train_t.std()
        X_train_t = (X_train_t - mean) / (std + 1e-6)
        X_test_t = (X_test_t - mean) / (std + 1e-6)

        y_train_t = torch.tensor(y_train_enc, dtype=torch.float32)
        y_test_t = torch.tensor(y_test_enc, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
