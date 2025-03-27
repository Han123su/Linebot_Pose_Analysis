import pywt
import numpy as np

def wavelet_filter(signal, wavelet='db5', level=6):
    """
    使用離散小波轉換 (DWT) 進行去雜訊濾波。
    
    Args:
        signal (np.ndarray): 一維數據。
        wavelet (str): 使用的小波母函數，預設為 'db5'。
        level (int): 分解層數，預設為 6。
        
    Returns:
        np.ndarray: 濾波後的信號。
    """
    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # 將高頻細節係數設為 0（從第 2 層到最後）
    for i in range(1, len(coeffs)):
        coeffs[i] = np.zeros_like(coeffs[i])
    
    # 重建信號（只保留 approximation 部分）
    filtered_signal = pywt.waverec(coeffs, wavelet)
    
    # 截斷至原始長度
    return filtered_signal[:len(signal)]

def perform_fft(signal, fs):
    """
    計算傅立葉轉換的功率頻譜。
    
    Args:
        signal (np.ndarray): 時域信號。
        fs (int or float): 取樣頻率。
    
    Returns:
        freqs (np.ndarray): 頻率軸。
        power (np.ndarray): 功率值。
    """
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_values = np.fft.rfft(signal)
    power = np.abs(fft_values) ** 2 / n
    return freqs, power

def calculate_approximate_entropy(U, m, r):
    """
    Approximate Entropy (ApEn) 計算。
    
    Args:
        U (np.ndarray): 訊號。
        m (int): 嵌入維度。
        r (float): 容差範圍，通常是 std(U) 的某個比例。
    
    Returns:
        float: Approximate entropy 值。
    """
    def _phi(m):
        N = len(U)
        x = np.array([U[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) / (N - m + 1)
        return np.sum(np.log(C)) / (N - m + 1)

    return abs(_phi(m) - _phi(m + 1))

