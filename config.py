# config.py
import os
import torch

# 基础配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_STATE = 42
RESULTS_DIR = "out"
PARAMS_FILE = "best_params.json"  # 最佳参数保存文件

# 确保结果目录存在
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# 数据路径配置 
# 修改 DATA_FILES 路径，指向 .txt 文件位置
DATA_FILES = {
    "class_1": r"data/1.txt",
    "class_2": r"data/2.txt",
    "class_3": r"data/3.txt",
    "class_4": r"data/4.txt",
    "class_5": r"data/5.txt",
}

# 固定参数
SIGNAL_LENGTH = 2500
TEST_SIZE = 0.2
NUM_EPOCHS_OPTUNA = 10   # 搜索时的训练轮数（少一点以节省时间）
NUM_EPOCHS_FINAL = 100   # 最终训练的轮数
N_TRIALS = 20            # Optuna 尝试的次数
SKIP_ROWS = 3            # 肌电仪器（BL-420N）输出的TXT文件的表头行数
