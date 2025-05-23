import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, windows
import pywt
from scipy.spatial import ConvexHull

def analyze_side_gait(xlsx_path, output_folder=None, fs=30):
    """
    分析側面步態數據，對應 Wavelet_filter.m 的功能
    
    參數:
        xlsx_path: Excel文件路徑
        output_folder: 輸出文件夾（此版本不使用）
        fs: 採樣頻率 (fps)
    
    返回:
        結果字符串
    """
    # 讀取Excel文件
    df = pd.read_excel(xlsx_path)
    
    # 提取關節角度數據
    L_knee_angle = df["Angle_23_25_27"].to_numpy()
    R_knee_angle = df["Angle_24_26_28"].to_numpy()
    L_hip_angle = df["Angle_11_23_25"].to_numpy()
    R_hip_angle = df["Angle_12_24_26"].to_numpy()
    L_ankle_angle = df["Angle_23_27_31"].to_numpy()
    R_ankle_angle = df["Angle_24_28_32"].to_numpy()
    
    # 小波分解和重建
    wavelet_function = 'coif3'
    level = 5
    
    # 關節角度處理
    L_knee_wavelet, _ = wavelet_process(L_knee_angle, wavelet_function, level, fs)
    R_knee_wavelet, _ = wavelet_process(R_knee_angle, wavelet_function, level, fs)
    L_hip_wavelet, _ = wavelet_process(L_hip_angle, wavelet_function, level, fs)
    R_hip_wavelet, _ = wavelet_process(R_hip_angle, wavelet_function, level, fs)
    L_ankle_wavelet, _ = wavelet_process(L_ankle_angle, wavelet_function, level, fs)
    R_ankle_wavelet, _ = wavelet_process(R_ankle_angle, wavelet_function, level, fs)

    # 自適應頻率分析
    freq_L_knee, amp_L_knee = perform_fft(L_knee_wavelet, fs)
    freq_R_knee, amp_R_knee = perform_fft(R_knee_wavelet, fs)
    freq_L_hip, amp_L_hip = perform_fft(L_hip_wavelet, fs)
    freq_R_hip, amp_R_hip = perform_fft(R_hip_wavelet, fs)
    freq_L_ankle, amp_L_ankle = perform_fft(L_ankle_wavelet, fs)
    freq_R_ankle, amp_R_ankle = perform_fft(R_ankle_wavelet, fs)
    
    # 設定最小有效頻率（排除直流分量和極低頻）
    min_valid_freq = 0.2  # Hz
    
    # 獲取主要頻率分量
    top_freq_L_knee = get_top_frequencies(freq_L_knee, amp_L_knee, min_valid_freq)
    top_freq_R_knee = get_top_frequencies(freq_R_knee, amp_R_knee, min_valid_freq)
    top_freq_L_hip = get_top_frequencies(freq_L_hip, amp_L_hip, min_valid_freq)
    top_freq_R_hip = get_top_frequencies(freq_R_hip, amp_R_hip, min_valid_freq)
    top_freq_L_ankle = get_top_frequencies(freq_L_ankle, amp_L_ankle, min_valid_freq)
    top_freq_R_ankle = get_top_frequencies(freq_R_ankle, amp_R_ankle, min_valid_freq)

    # 計算步行頻率
    knee_gait_freq = 1.0  # 預設值
    if len(top_freq_L_knee) > 0 and len(top_freq_R_knee) > 0:
        knee_gait_freq = np.mean([top_freq_L_knee[0], top_freq_R_knee[0]])
    
    hip_gait_freq = 1.0  # 預設值
    if len(top_freq_L_hip) > 0 and len(top_freq_R_hip) > 0:
        hip_gait_freq = np.mean([top_freq_L_hip[0], top_freq_R_hip[0]])

    ankle_gait_freq = 1.0    # 預設值
    if len(top_freq_L_ankle) > 0 and len(top_freq_R_ankle) > 0:
        ankle_gait_freq = np.mean([top_freq_L_ankle[0], top_freq_R_ankle[0]]) 
    
    # 使用膝關節和髖關節頻率的平均值作為整體步行頻率
    gait_freq = np.mean([knee_gait_freq, hip_gait_freq])
    
    # 設定截止頻率為步頻倍數
    cutoff_freq = gait_freq * 2.75
    
    # 設計濾波器 (Butterworth 低通濾波器)
    b, a = butter(4, cutoff_freq/(fs/2), 'low')
    
    # 應用零相位濾波到小波重建後的信號
    L_knee_denoised = filtfilt(b, a, L_knee_wavelet.flatten())
    R_knee_denoised = filtfilt(b, a, R_knee_wavelet.flatten())
    L_hip_denoised = filtfilt(b, a, L_hip_wavelet.flatten())
    R_hip_denoised = filtfilt(b, a, R_hip_wavelet.flatten())
    L_ankle_denoised = filtfilt(b, a, L_ankle_wavelet.flatten())
    R_ankle_denoised = filtfilt(b, a, R_ankle_wavelet.flatten())
    
    # 最終處理後的信號
    L_knee_aligned = L_knee_denoised
    R_knee_aligned = R_knee_denoised
    L_hip_aligned = L_hip_denoised
    R_hip_aligned = R_hip_denoised
    L_ankle_aligned = L_ankle_denoised
    R_ankle_aligned = R_ankle_denoised
    
    # 膝關節峰值檢測
    min_peak_height_knee = np.mean(L_knee_aligned)  # 用平均值作為最小峰值高度
    min_peak_distance = 30  # 設最小週期
    
    # 膝關節峰值檢測
    locs_L_knee = find_peaks(L_knee_aligned, height=min_peak_height_knee, 
                            distance=min_peak_distance)[0]
    locs_R_knee = find_peaks(R_knee_aligned, height=min_peak_height_knee, 
                            distance=min_peak_distance)[0]
    
    # 計算膝關節步態週期
    L_knee_period = np.nan
    if len(locs_L_knee) > 1:
        L_knee_period = np.mean(np.diff(locs_L_knee))
    
    R_knee_period = np.nan
    if len(locs_R_knee) > 1:
        R_knee_period = np.mean(np.diff(locs_R_knee))
    
    # 髖關節峰值檢測
    min_peak_height_hip = np.mean(L_hip_aligned)
    locs_L_hip = find_peaks(L_hip_aligned, height=min_peak_height_hip, 
                           distance=min_peak_distance)[0]
    locs_R_hip = find_peaks(R_hip_aligned, height=min_peak_height_hip, 
                           distance=min_peak_distance)[0]
    
    # 計算髖關節步態週期
    L_hip_period = np.nan
    if len(locs_L_hip) > 1:
        L_hip_period = np.mean(np.diff(locs_L_hip))
    
    R_hip_period = np.nan
    if len(locs_R_hip) > 1:
        R_hip_period = np.mean(np.diff(locs_R_hip))
    
    average_knee_period = np.nanmean([L_knee_period, R_knee_period])
    
    # 計算速度和加速度
    L_knee_velocity = np.gradient(L_knee_aligned)
    R_knee_velocity = np.gradient(R_knee_aligned)
    L_knee_acceleration = np.gradient(L_knee_velocity)
    R_knee_acceleration = np.gradient(R_knee_velocity)
    
    L_hip_velocity = np.gradient(L_hip_aligned)
    R_hip_velocity = np.gradient(R_hip_aligned)
    L_hip_acceleration = np.gradient(L_hip_velocity)
    R_hip_acceleration = np.gradient(R_hip_velocity)
    
    # 計算角速度的絕對值平均
    L_knee_vel_magnitude = np.mean(np.abs(L_knee_velocity))
    R_knee_vel_magnitude = np.mean(np.abs(R_knee_velocity))
    
    # 計算加速度的絕對值平均
    L_knee_acc_magnitude = np.mean(np.abs(L_knee_acceleration))
    R_knee_acc_magnitude = np.mean(np.abs(R_knee_acceleration))
    
    # 計算髖關節的運動特徵
    L_hip_vel_magnitude = np.mean(np.abs(L_hip_velocity))
    R_hip_vel_magnitude = np.mean(np.abs(R_hip_velocity))
    
    L_hip_acc_magnitude = np.mean(np.abs(L_hip_acceleration))
    R_hip_acc_magnitude = np.mean(np.abs(R_hip_acceleration))

    # 踝關節相關
    min_peak_height_ankle = np.mean([np.mean(L_ankle_aligned), np.mean(R_ankle_aligned)]) * 0.9
    locs_L_ankle = find_peaks(L_ankle_aligned, height=min_peak_height_ankle, distance=min_peak_distance)[0]
    locs_R_ankle = find_peaks(R_ankle_aligned, height=min_peak_height_ankle, distance=min_peak_distance)[0]

    L_ankle_period = np.mean(np.diff(locs_L_ankle)) if len(locs_L_ankle) > 1 else np.nan
    R_ankle_period = np.mean(np.diff(locs_R_ankle)) if len(locs_R_ankle) > 1 else np.nan

    # 多週期踝關節角度範圍分析
    def cycle_range_analysis(signal, locs):
        ranges = []
        if len(locs) > 1:
            for i in range(len(locs)-1):
                seg = signal[locs[i]:locs[i+1]]
                ranges.append(np.max(seg) - np.min(seg))
        return np.mean(ranges) if ranges else np.nan, np.std(ranges) if ranges else np.nan

    L_ankle_mean_cycle_range, L_ankle_std_cycle_range = cycle_range_analysis(L_ankle_aligned, locs_L_ankle)
    R_ankle_mean_cycle_range, R_ankle_std_cycle_range = cycle_range_analysis(R_ankle_aligned, locs_R_ankle)

    ankle_mean_cycle_range_diff = abs(L_ankle_mean_cycle_range - R_ankle_mean_cycle_range) if not np.isnan(L_ankle_mean_cycle_range) and not np.isnan(R_ankle_mean_cycle_range) else np.nan

    # 速度與加速度
    L_ankle_velocity = np.gradient(L_ankle_aligned)
    R_ankle_velocity = np.gradient(R_ankle_aligned)
    L_ankle_acceleration = np.gradient(L_ankle_velocity)
    R_ankle_acceleration = np.gradient(R_ankle_velocity)

    L_ankle_vel_magnitude = np.mean(np.abs(L_ankle_velocity))
    R_ankle_vel_magnitude = np.mean(np.abs(R_ankle_velocity))
    L_ankle_acc_magnitude = np.mean(np.abs(L_ankle_acceleration))
    R_ankle_acc_magnitude = np.mean(np.abs(R_ankle_acceleration))
    
    # 多周期角度範圍分析
    # 膝關節多周期範圍分析
    L_knee_cycle_ranges = []
    R_knee_cycle_ranges = []
    L_knee_mean_cycle_range = np.nan
    L_knee_std_cycle_range = np.nan
    R_knee_mean_cycle_range = np.nan
    R_knee_std_cycle_range = np.nan
    
    # 左膝周期範圍分析
    if len(locs_L_knee) > 1:
        for i in range(len(locs_L_knee)-1):
            cycle_start = locs_L_knee[i]
            cycle_end = locs_L_knee[i+1]
            cycle_data = L_knee_aligned[cycle_start:cycle_end]
            cycle_range = np.max(cycle_data) - np.min(cycle_data)
            L_knee_cycle_ranges.append(cycle_range)
        L_knee_mean_cycle_range = np.mean(L_knee_cycle_ranges)
        L_knee_std_cycle_range = np.std(L_knee_cycle_ranges)
    
    # 右膝周期範圍分析
    if len(locs_R_knee) > 1:
        for i in range(len(locs_R_knee)-1):
            cycle_start = locs_R_knee[i]
            cycle_end = locs_R_knee[i+1]
            cycle_data = R_knee_aligned[cycle_start:cycle_end]
            cycle_range = np.max(cycle_data) - np.min(cycle_data)
            R_knee_cycle_ranges.append(cycle_range)
        R_knee_mean_cycle_range = np.mean(R_knee_cycle_ranges)
        R_knee_std_cycle_range = np.std(R_knee_cycle_ranges)
    
    # 髖關節多周期範圍分析
    L_hip_cycle_ranges = []
    R_hip_cycle_ranges = []
    L_hip_mean_cycle_range = np.nan
    L_hip_std_cycle_range = np.nan
    R_hip_mean_cycle_range = np.nan
    R_hip_std_cycle_range = np.nan
    
    # 左髖周期範圍分析
    if len(locs_L_hip) > 1:
        for i in range(len(locs_L_hip)-1):
            cycle_start = locs_L_hip[i]
            cycle_end = locs_L_hip[i+1]
            cycle_data = L_hip_aligned[cycle_start:cycle_end]
            cycle_range = np.max(cycle_data) - np.min(cycle_data)
            L_hip_cycle_ranges.append(cycle_range)
        L_hip_mean_cycle_range = np.mean(L_hip_cycle_ranges)
        L_hip_std_cycle_range = np.std(L_hip_cycle_ranges)
    
    # 右髖周期範圍分析
    if len(locs_R_hip) > 1:
        for i in range(len(locs_R_hip)-1):
            cycle_start = locs_R_hip[i]
            cycle_end = locs_R_hip[i+1]
            cycle_data = R_hip_aligned[cycle_start:cycle_end]
            cycle_range = np.max(cycle_data) - np.min(cycle_data)
            R_hip_cycle_ranges.append(cycle_range)
        R_hip_mean_cycle_range = np.mean(R_hip_cycle_ranges)
        R_hip_std_cycle_range = np.std(R_hip_cycle_ranges)
    
    # 計算波形參數比較
    knee_waveform_sim = np.nan
    hip_waveform_sim = np.nan
    knee_amplitude_diff = np.nan
    knee_interval_diff = np.nan
    hip_amplitude_diff = np.nan
    hip_interval_diff = np.nan
    
    if not np.isnan(L_knee_period) and not np.isnan(R_knee_period) and not np.isnan(L_knee_mean_cycle_range) and not np.isnan(R_knee_mean_cycle_range):
        knee_amplitude_diff = abs(L_knee_mean_cycle_range - R_knee_mean_cycle_range) / L_knee_mean_cycle_range
        knee_interval_diff = abs(L_knee_period - R_knee_period) / L_knee_period
        weight_amplitude = 0.6  # 振幅權重60%
        weight_interval = 0.4   # 間隔權重40%
        knee_waveform_sim = 1 - (knee_amplitude_diff * weight_amplitude + knee_interval_diff * weight_interval)
    
    if not np.isnan(L_hip_period) and not np.isnan(R_hip_period) and not np.isnan(L_hip_mean_cycle_range) and not np.isnan(R_hip_mean_cycle_range):
        hip_amplitude_diff = abs(L_hip_mean_cycle_range - R_hip_mean_cycle_range) / L_hip_mean_cycle_range
        hip_interval_diff = abs(L_hip_period - R_hip_period) / L_hip_period
        hip_waveform_sim = 1 - ((hip_amplitude_diff + hip_interval_diff) / 2)
    
    if not np.isnan(L_ankle_period) and not np.isnan(R_ankle_period) and not np.isnan(L_ankle_mean_cycle_range) and not np.isnan(R_ankle_mean_cycle_range):
        amp_diff = abs(L_ankle_mean_cycle_range - R_ankle_mean_cycle_range) / L_ankle_mean_cycle_range
        intv_diff = abs(L_ankle_period - R_ankle_period) / L_ankle_period
        ankle_waveform_sim = 1 - (amp_diff * 0.6 + intv_diff * 0.4)
    else:
        ankle_waveform_sim = np.nan

    # 熵值分析
    L_knee_entropy = approximate_entropy(L_knee_aligned)
    R_knee_entropy = approximate_entropy(R_knee_aligned)
    L_hip_entropy = approximate_entropy(L_hip_aligned)
    R_hip_entropy = approximate_entropy(R_hip_aligned)
    L_ankle_entropy = approximate_entropy(L_ankle_aligned)
    R_ankle_entropy = approximate_entropy(R_ankle_aligned)
    
    if not np.isnan(average_knee_period):
        result = f"平均步態週期: {average_knee_period:.2f} frames\n\n"
    else:
        result += "無法計算平均步態週期\n\n"
    
    # 關節運動特徵
    result += "===== 關節運動特徵 =====\n"
    result += f"膝角速度標準差 - 左:{np.std(L_knee_velocity):.2f} 右:{np.std(R_knee_velocity):.2f}\n"
    result += f"膝加速度標準差 - 左:{np.std(L_knee_acceleration):.2f} 右:{np.std(R_knee_acceleration):.2f}\n"
    result += f"膝平均角速度大小 - 左:{L_knee_vel_magnitude:.2f} 右:{R_knee_vel_magnitude:.2f}\n\n"
    
    result += f"髖角速度標準差 - 左:{np.std(L_hip_velocity):.2f} 右:{np.std(R_hip_velocity):.2f}\n"
    result += f"髖加速度標準差 - 左:{np.std(L_hip_acceleration):.2f} 右:{np.std(R_hip_acceleration):.2f}\n"
    result += f"髖平均角速度大小 - 左:{L_hip_vel_magnitude:.2f} 右:{R_hip_vel_magnitude:.2f}\n\n"
    
    result += f"踝角速度標準差 - 左:{np.std(L_ankle_velocity):.2f} 右:{np.std(R_ankle_velocity):.2f}\n"
    result += f"踝加速度標準差 - 左:{np.std(L_ankle_acceleration):.2f} 右:{np.std(R_ankle_acceleration):.2f}\n"
    result += f"踝平均角速度大小 - 左:{L_ankle_vel_magnitude:.2f} 右:{R_ankle_vel_magnitude:.2f}\n\n"
    
    # 多周期關節角度範圍分析
    result += "==== 週期角度範圍分析 ====\n"
    
    if not np.isnan(L_knee_mean_cycle_range):
        result += f"左膝平均週期範圍: {L_knee_mean_cycle_range:.2f} ± {L_knee_std_cycle_range:.2f}\n"
    else:
        result += "左膝未檢測到足夠的週期點\n"
    
    if not np.isnan(R_knee_mean_cycle_range):
        result += f"右膝平均週期範圍: {R_knee_mean_cycle_range:.2f} ± {R_knee_std_cycle_range:.2f}\n"
    else:
        result += "右膝未檢測到足夠的週期點\n"
    
    if not np.isnan(L_knee_mean_cycle_range) and not np.isnan(R_knee_mean_cycle_range):
        knee_mean_cycle_range_diff = abs(L_knee_mean_cycle_range - R_knee_mean_cycle_range)
        result += f"膝左右平均範圍差異: {knee_mean_cycle_range_diff:.2f}\n\n"
    else:
        result += "無法計算膝關節左右平均週期範圍差異\n\n"
    
    if not np.isnan(L_hip_mean_cycle_range):
        result += f"左髖平均週期範圍: {L_hip_mean_cycle_range:.2f} ± {L_hip_std_cycle_range:.2f}\n"
    else:
        result += "左髖未檢測到足夠的週期點\n"
    
    if not np.isnan(R_hip_mean_cycle_range):
        result += f"右髖平均週期範圍: {R_hip_mean_cycle_range:.2f} ± {R_hip_std_cycle_range:.2f}\n"
    else:
        result += "右髖未檢測到足夠的週期點\n"
    
    if not np.isnan(L_hip_mean_cycle_range) and not np.isnan(R_hip_mean_cycle_range):
        hip_mean_cycle_range_diff = abs(L_hip_mean_cycle_range - R_hip_mean_cycle_range)
        result += f"髖左右平均範圍差異: {hip_mean_cycle_range_diff:.2f}\n\n"
    else:
        result += "無法計算髖關節左右平均週期範圍差異\n"
    
    if not np.isnan(L_ankle_mean_cycle_range):
        result += f"左踝平均週期範圍: {L_ankle_mean_cycle_range:.2f} ± {L_ankle_std_cycle_range:.2f}\n"
    else:
        result += "左踝未檢測到足夠的週期點\n"
    if not np.isnan(R_ankle_mean_cycle_range):
        result += f"右踝平均週期範圍: {R_ankle_mean_cycle_range:.2f} ± {R_ankle_std_cycle_range:.2f}\n"
    else:
        result += "右踝未檢測到足夠的週期點\n"
    if not np.isnan(ankle_mean_cycle_range_diff):
        result += f"踝左右平均範圍差異: {ankle_mean_cycle_range_diff:.2f}\n\n"
    else:
        result += "無法計算踝關節左右平均週期範圍差異\n\n"

    # 熵值分析
    result += "===== 熵值分析 =====\n"
    result += f"膝近似熵 - 左:{L_knee_entropy:.4f} 右:{R_knee_entropy:.4f}\n"
    result += f"膝近似熵差異 - {abs(L_knee_entropy - R_knee_entropy):.4f}\n"
    
    result += f"髖近似熵 - 左:{L_hip_entropy:.4f} 右:{R_hip_entropy:.4f}\n"
    result += f"髖近似熵差異 - {abs(L_hip_entropy - R_hip_entropy):.4f}\n"

    result += f"踝近似熵 - 左:{L_ankle_entropy:.4f} 右:{R_ankle_entropy:.4f}\n"
    result += f"踝近似熵差異 - {abs(L_ankle_entropy - R_ankle_entropy):.4f}\n\n"
    
    # 波形參數比較分析
    result += "===== 波形參數比較分析 =====\n"
    if not np.isnan(knee_waveform_sim):
        result += f"膝關節波形參數相似度: {knee_waveform_sim:.4f}\n"
    else:
        result += "膝關節檢測不到足夠的波峰來分析波形參數\n"
    
    if not np.isnan(hip_waveform_sim):
        result += f"髖關節波形參數相似度: {hip_waveform_sim:.4f}\n"
    else:
        result += "髖關節檢測不到足夠的波峰來分析波形參數\n"
    
    if not np.isnan(ankle_waveform_sim):
        result += f"踝關節波形參數相似度: {ankle_waveform_sim:.4f}\n\n"
    else:
        result += "踝關節檢測不到足夠的波峰來分析波形參數\n\n"

    # === 擺動分析 ===
    result += "===== 擺動分析 =====\n"

    def do_pca(data):
        data_centered = data - np.mean(data, axis=0)
        U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
        coeff = Vt.T
        score = data_centered @ coeff
        latent = (S**2) / (data.shape[0] - 1)
        return coeff, score, latent

    def angle_range(vectors):
        angles = np.mod(np.degrees(np.arctan2(vectors[:,1], vectors[:,0])), 360)
        sorted_angles = np.sort(angles)
        diffs = np.diff(np.concatenate([sorted_angles, [sorted_angles[0] + 360]]))
        return 360 - np.max(diffs)

    # 【hip-knee】方向角範圍
    L_hip_knee_vec = np.stack([df['L_Knee_25_x'] - df['L_Hip_23_x'], df['L_Knee_25_y'] - df['L_Hip_23_y']], axis=1)
    R_hip_knee_vec = np.stack([df['R_Knee_26_x'] - df['R_Hip_24_x'], df['R_Knee_26_y'] - df['R_Hip_24_y']], axis=1)

    angle_range_L = angle_range(L_hip_knee_vec)
    angle_range_R = angle_range(R_hip_knee_vec)
    diff_range = abs(angle_range_L - angle_range_R)

    result += f"肩-髖 方向角範圍：\n左:{angle_range_L:.2f}° 右:{angle_range_R:.2f}° 差異:{diff_range:.2f}°\n"

    # 【膝關節】路徑長度與主軸角度
    L_knee_points = np.stack([df['L_Knee_25_x'], df['L_Knee_25_y']], axis=1)
    R_knee_points = np.stack([df['R_Knee_26_x'], df['R_Knee_26_y']], axis=1)

    def pca_path_analysis(points):
        mean_centered = points - np.mean(points, axis=0)
        cov_inv = np.linalg.pinv(np.cov(mean_centered.T))
        dists = np.sqrt(np.sum((mean_centered @ cov_inv) * mean_centered, axis=1))
        inliers = dists <= np.sqrt(5.99)
        clean = points[inliers]
        path_len = np.sum(np.linalg.norm(np.diff(clean, axis=0), axis=1))
        coeff, _, _ = do_pca(clean)
        return path_len, coeff

    # 執行 PCA 分析
    path_L, coeff_L = pca_path_analysis(L_knee_points)
    path_R, coeff_R = pca_path_analysis(R_knee_points)

    # 對齊主軸方向
    if np.sign(coeff_L[0, 0]) != np.sign(coeff_R[0, 0]):
        coeff_R[:, 0] = -coeff_R[:, 0]

    def principal_axis_angle(coeff):
        """轉換主軸向量的角度，使其位於 [-90°, +90°] 區間內，利於比較方向性"""
        angle = np.degrees(np.arctan2(coeff[1, 0], coeff[0, 0]))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        return angle

    # 使用方向標準化計算角度
    angle_L = principal_axis_angle(coeff_L)
    angle_R = principal_axis_angle(coeff_R)
    angle_diff = abs(abs(angle_L) - abs(angle_R))

    result += f"膝關節 路徑長度：\n左:{path_L:.2f} 右:{path_R:.2f} \n差異:{abs(path_L - path_R):.2f}\n"
    result += f"膝關節 主軸角度：\n左:{angle_L:.2f}° 右:{angle_R:.2f}° 差異:{angle_diff:.2f}°\n\n"

    # 計算腳部三角形重心座標（根據 ankle, heel, foot）
    centroid_R_dense = np.stack([
        (df['R_Ankle_28_x'] + df['R_Heel_30_x'] + df['R_Foot_32_x']) / 3,
        (df['R_Ankle_28_y'] + df['R_Heel_30_y'] + df['R_Foot_32_y']) / 3
    ], axis=1)

    centroid_L_dense = np.stack([
        (df['L_Ankle_27_x'] + df['L_Heel_29_x'] + df['L_Foot_31_x']) / 3,
        (df['L_Ankle_27_y'] + df['L_Heel_29_y'] + df['L_Foot_31_y']) / 3
    ], axis=1)

    # def classify_gait_phases(heel_y, toe_y, centroid, angle):
    #     # 建立凸包9區域門檻
    #     cx = centroid[:, 0]
    #     cy = centroid[:, 1]
    #     minx, maxx = np.min(cx), np.max(cx)
    #     miny, maxy = np.min(cy), np.max(cy)
    #     x1, x2 = minx + (maxx - minx) / 3, minx + 2 * (maxx - minx) / 3
    #     y1, y2 = miny + (maxy - miny) / 3, miny + 2 * (maxy - miny) / 3

    #     def region(x, y):
    #         col = 1 if x < x1 else (3 if x > x2 else 2)
    #         row = 1 if y < y1 else (3 if y > y2 else 2)
    #         return (row - 1) * 3 + col

    #     dy = heel_y - toe_y
    #     n = len(dy)
    #     phase = np.zeros(n, dtype=int)
    #     prev = 4

    #     for i in range(n):
    #         r = region(cx[i], cy[i])
    #         d = dy[i]
    #         a = angle[i]
    #         p = prev

    #         if r == 2:
    #             p = 4
    #         elif r == 5 or r == 8:
    #             p = 2
    #         elif r == 4:
    #             p = 1 if d > 2 else 2
    #         elif r == 6:
    #             p = 3 if d < -2 else 2
    #         elif r == 1:
    #             if d > 2 and a < -40:
    #                 p = 1
    #             else:
    #                 p = 2
    #         elif r == 3:
    #             if d < -2 and a > 40:
    #                 p = 3
    #             else:
    #                 p = 2
    #         elif r == 7:
    #             p = 1
    #         elif r == 9:
    #             p = 3
    #         else:
    #             p = prev

    #         if (prev == 1 and p == 2) or (prev == 2 and p == 3) or (prev == 3 and p == 4) or (prev == 4 and p == 1):
    #             phase[i] = p
    #             prev = p
    #         else:
    #             phase[i] = prev

    #     stats = [np.sum(phase == k) / n * 100 for k in [1, 2, 3, 4]]
    #     return stats

    # heel_R_y = df['R_Heel_30_y'].to_numpy()
    # toe_R_y = df['R_Foot_32_y'].to_numpy()
    # angle_R = df['Angle_28_30_32'].to_numpy()

    # heel_L_y = df['L_Heel_29_y'].to_numpy()
    # toe_L_y = df['L_Foot_31_y'].to_numpy()
    # angle_L = df['Angle_27_29_31'].to_numpy()

    # phase_stats_R = classify_gait_phases(heel_R_y, toe_R_y, centroid_R_dense, angle_R)
    # phase_stats_L = classify_gait_phases(heel_L_y, toe_L_y, centroid_L_dense, angle_L)

    # === 凸包活動區面積、長寬比、步態階段比例 ===
    result += "=== 凸包活動區面積比分析 ===\n"
    hull_R = ConvexHull(centroid_R_dense)
    area_R = hull_R.volume
    hull_L = ConvexHull(centroid_L_dense)
    area_L = hull_L.volume
    result += f"右腳面積: {area_R:.2f} px^2\n"
    result += f"左腳面積: {area_L:.2f} px^2\n"
    result += f"凸包面積比: {area_R / area_L:.2f}\n"

    result += "\n=== 軌跡方向性(長寬比)分析 ===\n"
    xR, yR = centroid_R_dense[:,0], centroid_R_dense[:,1]
    xL, yL = centroid_L_dense[:,0], centroid_L_dense[:,1]
    width_R = xR.max() - xR.min()
    height_R = yR.max() - yR.min()
    ar_R = width_R / height_R
    width_L = xL.max() - xL.min()
    height_L = yL.max() - yL.min()
    ar_L = width_L / height_L
    result += f"右腳 長寬比: {ar_R:.2f}\n"
    result += f"左腳 長寬比: {ar_L:.2f}\n"
    result += f"差值: {abs(ar_R - ar_L):.2f}\n"

    # result += "\n===== 右腳階段時間分配分析 =====\n"
    # result += f"  腳跟著地: {phase_stats_R[0]:.1f}%\n"
    # result += f"  整腳著地: {phase_stats_R[1]:.1f}%\n"
    # result += f"  腳跟離地: {phase_stats_R[2]:.1f}%\n"
    # result += f"  整腳離地: {phase_stats_R[3]:.1f}%\n"

    # result += "\n===== 左腳階段時間分配分析 =====\n"
    # result += f"  腳跟著地: {phase_stats_L[0]:.1f}%\n"
    # result += f"  整腳著地: {phase_stats_L[1]:.1f}%\n"
    # result += f"  腳跟離地: {phase_stats_L[2]:.1f}%\n"
    # result += f"  整腳離地: {phase_stats_L[3]:.1f}%\n"

    return result

def wrcoef_custom(coeffs, level, wavelet, part='a'):
    N = len(coeffs)
    rec_coeffs = [np.zeros_like(c) for c in coeffs]  # 預設全部為0
    
    if part == 'a':
        if level == N - 1:
            rec_coeffs[0] = coeffs[0]
        else:
            raise ValueError(f"無法取得第 {level} 層 approximation（可用最大層數為 {N-1}）")
    
    elif part == 'd':
        target = N - level
        if 1 <= target < N:
            rec_coeffs[target] = coeffs[target]
        else:
            raise ValueError(f"無法取得第 {level} 層 detail（可用最大層數為 {N-1}）")
    
    else:
        raise ValueError("part 必須是 'a' 或 'd'")
    
    return pywt.waverec(rec_coeffs, wavelet)


def wavelet_process(signal, wavelet, level, fs):
    """仿照 MATLAB 重建 A5, D5, D4 並組合"""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    A5 = wrcoef_custom(coeffs, 5, wavelet, 'a')
    D5 = wrcoef_custom(coeffs, 5, wavelet, 'd')
    D4 = wrcoef_custom(coeffs, 4, wavelet, 'd')
    
    # 平滑 A5
    window_size = int(fs * 0.5)
    A5_smooth = np.convolve(A5, np.ones(window_size)/window_size, mode='same')
    
    return D4 + D5 + A5_smooth, A5_smooth

def perform_fft(signal, fs):
    signal = np.asarray(signal).flatten() 
    signal = signal - np.mean(signal)

    window = windows.hann(len(signal))
    windowed_signal = signal * window

    N = len(signal)
    Y = np.fft.fft(windowed_signal)
    Y = Y[:N//2+1]

    freq = np.arange(len(Y)) * fs / N
    amplitude = np.abs(Y)/N * 2
    amplitude[0] = amplitude[0] / 2

    return freq, amplitude

def get_top_frequencies(freq, amp, min_freq=0.2, top_n=3):
    """獲取主要頻率分量"""
    valid_idx = np.where(freq >= min_freq)[0]
    if len(valid_idx) == 0:
        return []
    
    # 按振幅排序
    sorted_idx = np.argsort(amp[valid_idx])[::-1]
    global_idx = valid_idx[sorted_idx]
    
    # 返回前n個頻率
    top_n = min(top_n, len(global_idx))
    return freq[global_idx[:top_n]].tolist()

def approximate_entropy(data, m=2, r=0.2):
    """近似熵計算，對應 approximate_entropy.m"""
    N = len(data)
    
    # 數據標準化
    if np.std(data) != 0:
        r = r * np.std(data)
    else:
        r = 0.2  # 默認值
    
    def phi(m_val):
        count = 0
        total = 0
        
        # 建立嵌入向量
        patterns = []
        for i in range(N - m_val + 1):
            patterns.append(data[i:i + m_val])
        
        patterns = np.array(patterns)
        
        # 計算相似度
        for i in range(N - m_val + 1):
            # 計算當前模式與所有模式之間的最大距離
            distances = np.max(np.abs(patterns - patterns[i]), axis=1)
            
            # 計算距離小於r的模式數量
            count = np.sum(distances <= r)
            
            # 計算對數求和
            if count > 0:
                total += np.log(count / (N - m_val + 1))
        
        # 返回phi值
        if N - m_val + 1 > 0:
            return total / (N - m_val + 1)
        else:
            return 0
    
    # 計算近似熵
    return np.abs(phi(m) - phi(m+1))

    # === 擺動分析 ===
    result += "===== 擺動分析 =====\n"
    def do_pca(data):
        data_centered = data - np.mean(data, axis=0)
        U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
        coeff = Vt.T
        score = data_centered @ coeff
        latent = (S**2) / (data.shape[0] - 1)
        return coeff, score, latent

    def angle_range(vectors):
        angles = np.mod(np.degrees(np.arctan2(vectors[:,1], vectors[:,0])), 360)
        sorted_angles = np.sort(angles)
        diffs = np.diff(np.concatenate([sorted_angles, [sorted_angles[0] + 360]]))
        return 360 - np.max(diffs)

    # 【hip-knee】方向角範圍
    L_hip_knee_vec = np.stack([df['L_Knee_25_x'] - df['L_Hip_23_x'], df['L_Knee_25_y'] - df['L_Hip_23_y']], axis=1)
    R_hip_knee_vec = np.stack([df['R_Knee_26_x'] - df['R_Hip_24_x'], df['R_Knee_26_y'] - df['R_Hip_24_y']], axis=1)

    angle_range_L = angle_range(L_hip_knee_vec)
    angle_range_R = angle_range(R_hip_knee_vec)
    diff_range = abs(angle_range_L - angle_range_R)

    result += f"【hip-knee】方向角範圍：左:{angle_range_L:.2f}° 右:{angle_range_R:.2f}° 差異:{diff_range:.2f}°\n"

    # 【膝關節】路徑長度與主軸角度
    L_knee_points = np.stack([df['L_Knee_25_x'], df['L_Knee_25_y']], axis=1)
    R_knee_points = np.stack([df['R_Knee_26_x'], df['R_Knee_26_y']], axis=1)

    def pca_path_analysis(points):
        mean_centered = points - np.mean(points, axis=0)
        cov_inv = np.linalg.pinv(np.cov(mean_centered.T))
        dists = np.sqrt(np.sum((mean_centered @ cov_inv) * mean_centered, axis=1))
        inliers = dists <= np.sqrt(5.99)
        clean = points[inliers]
        path_len = np.sum(np.linalg.norm(np.diff(clean, axis=0), axis=1))
        coeff, _, _ = do_pca(clean)
        angle = np.degrees(np.arctan2(coeff[1,0], coeff[0,0]))
        return path_len, angle, coeff

    path_L, angle_L, coeff_L = pca_path_analysis(L_knee_points)
    path_R, angle_R, coeff_R = pca_path_analysis(R_knee_points)

    if np.sign(coeff_L[0,0]) != np.sign(coeff_R[0,0]):
        coeff_R[:,0] = -coeff_R[:,0]
        angle_R = np.degrees(np.arctan2(coeff_R[1,0], coeff_R[0,0]))

    angle_diff = abs(abs(angle_L) - abs(angle_R))
    if angle_diff > 90:
        angle_diff = 180 - angle_diff

    result += f"【膝關節】路徑長度：左:{path_L:.2f} 右:{path_R:.2f} 差異:{abs(path_L - path_R):.2f}\n"
    result += f"【膝關節】主軸角度：左:{angle_L:.2f}° 右:{angle_R:.2f}° 差異:{angle_diff:.2f}°\n\n"

    return result
