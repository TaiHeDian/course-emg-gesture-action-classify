# models.py
import torch.nn as nn

class OptimizedCNN(nn.Module):
    def __init__(self, n_classes, n_filters=32, kernel_size=5, dropout=0.2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, n_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 2
            nn.Conv1d(n_filters, n_filters*2, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(n_filters*2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 3
            nn.Conv1d(n_filters*2, n_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_filters*4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16) # 强制压缩到固定长度，避免尺寸计算问题
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),  # 自动推断输入维度
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
