
import pandas as pd
import numpy as np
from side_analysis_modules import wavelet_filter, perform_fft, calculate_approximate_entropy

def analyze_side_main(xlsx_path, fs=30):
    df = pd.read_excel(xlsx_path)

    # 根據欄位名稱判斷是否為左側或右側
    has_left = 'Lknee_y' in df.columns
    has_right = 'Rknee_y' in df.columns

    results = {}

    if has_left:
        signal = df['Lknee_y'].values
        signal = wavelet_filter(signal)
        freqs, power = perform_fft(signal, fs)
        ap_en = calculate_approximate_entropy(signal, m=2, r=0.2 * np.std(signal))
        results['left'] = {
            'approx_entropy': ap_en,
            'dominant_freq': freqs[np.argmax(power)],
            'power_max': np.max(power)
        }

    if has_right:
        signal = df['Rknee_y'].values
        signal = wavelet_filter(signal)
        freqs, power = perform_fft(signal, fs)
        ap_en = calculate_approximate_entropy(signal, m=2, r=0.2 * np.std(signal))
        results['right'] = {
            'approx_entropy': ap_en,
            'dominant_freq': freqs[np.argmax(power)],
            'power_max': np.max(power)
        }

    # 輸出分析結果
    print("=== 側面影片分析結果 ===")
    if 'left' in results:
        print(f"左膝近似熵: {results['left']['approx_entropy']:.4f}")
        print(f"左膝主頻率: {results['left']['dominant_freq']:.2f} Hz")
        print(f"左膝最大功率: {results['left']['power_max']:.2f}")
    if 'right' in results:
        print(f"右膝近似熵: {results['right']['approx_entropy']:.4f}")
        print(f"右膝主頻率: {results['right']['dominant_freq']:.2f} Hz")
        print(f"右膝最大功率: {results['right']['power_max']:.2f}")

if __name__ == "__main__":
    analyze_side_main("Data_750g2.xlsx")
