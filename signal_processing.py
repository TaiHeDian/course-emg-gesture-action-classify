import numpy as np
import matplotlib.pyplot as plt

def compute_rms(signal: np.ndarray, window_size: int = 2500) -> np.ndarray:
    """
    计算信号的滑动 RMS（均方根）值。
    
    Parameters
    ----------
    signal : np.ndarray
        一维信号数组。
    window_size : int
        滑动窗口大小（样本数），例如 2500 对应 0.5s。
        
    Returns
    -------
    rms : np.ndarray
        每个窗口对应的 RMS 值数组，长度为 len(signal)-window_size+1。
    """
    # 先平方，再与全 1 卷积求平均，最后开根号
    kernel = np.ones(window_size) / window_size
    mean_sq = np.convolve(signal**2, kernel, mode='valid')
    return np.sqrt(mean_sq)


def load_signal(file_path: str, skip_header: int = 0) -> np.ndarray:
    """
    从文本文件加载一条 EMG 信号。假设每一行都是一个数值（前 skip_header 行为表头）。
    
    Parameters
    ----------
    file_path : str
        文本文件路径。
    skip_header : int
        跳过的表头行数。
        
    Returns
    -------
    signal : np.ndarray
        加载后的一维信号数组。
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # 跳过前几行表头，再把每行转成浮点数
    data_lines = lines[skip_header:]
    signal = np.array([float(line.strip()) for line in data_lines], dtype=np.float32)
    return signal


def sliding_window_analysis(signal: np.ndarray,
                            window_size: int = 2500,
                            step: int = 500,
                            threshold: float = 0.044):
    """
    在信号上做滑动窗口 RMS 分析，并根据阈值分类。
    
    Parameters
    ----------
    signal : np.ndarray
        一维信号数组。
    window_size : int
        窗口大小（样本数）。
    step : int
        窗口滑动步长（样本数）。
    threshold : float
        RMS 阈值，大于等于认为是“活动”。
    
    Returns
    -------
    results : dict
        包含以下字段：
        - 'rms'      : 所有窗口的 RMS 值列表
        - 'positions': 窗口起始位置列表（以窗口编号为单位）
        - 'active'   : 符合阈值的窗口编号列表
        - 'inactive' : 不符合阈值的窗口编号列表
    """
    n = len(signal)
    rms_list = []
    pos_list = []
    active, inactive = [], []
    
    idx = 0
    while idx + window_size <= n:
        window = signal[idx : idx + window_size]
        rms_val = compute_rms(window, window_size=window_size)[0]
        rms_list.append(rms_val)
        pos_list.append(idx // step)
        
        if rms_val >= threshold:
            active.append(idx // step)
        else:
            inactive.append(idx // step)
        
        idx += step
    
    return {
        'rms': rms_list,
        'positions': pos_list,
        'active': active,
        'inactive': inactive
    }


if __name__ == '__main__':
    # ---- 用户需配置 ----
    file_path    = r"data\1.txt"
    skip_header  = 3      # 前两行是表头
    window_size  = 2500   # 窗口大小（样本数）
    step         = 500    # 每次滑动 500 个样本
    threshold    = 0.044  # RMS 阈值
    # ---------------------
    
    # 1. 加载信号
    signal = load_signal(file_path, skip_header=skip_header)
    print(f"加载信号长度：{len(signal)}")
    
    # 2. 滑动窗口 RMS 分析
    res = sliding_window_analysis(signal,
                                  window_size=window_size,
                                  step=step,
                                  threshold=threshold)
    
    # 3. 绘制 RMS 曲线
    plt.figure(figsize=(10, 4))
    plt.plot(res['positions'], res['rms'])
    plt.axhline(threshold, color='r', linestyle='--', label=f"Threshold={threshold}")
    plt.xlabel('Window Index')
    plt.ylabel('RMS Value')
    plt.title('Sliding Window RMS Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 4. 输出分析结果
    print(f"总窗口数: {len(res['rms'])}")
    print(f"活动窗口 (RMS ≥ {threshold}) 个数: {len(res['active'])}")
    print(f"静默窗口 (RMS < {threshold}) 个数: {len(res['inactive'])}")
