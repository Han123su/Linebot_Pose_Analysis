import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def Lift_ratio(excel_file):
    # 創建結果文件夾
    output_folder = os.path.join(os.path.dirname(excel_file), 'image2')
    os.makedirs(output_folder, exist_ok=True)

    # 創建結果文本文件
    results_file_path = os.path.join(os.path.dirname(excel_file), 'lift_ratio_results.txt')
    with open(results_file_path, 'w') as results_file:
        results_file.write("============================\n")

        # 讀取 Excel 檔案
        df = pd.read_excel(excel_file, engine='openpyxl')

        # 計算每個特徵的值
        Lknee = 450 - df['L_Knee_25'].dropna().values
        Rknee = 450 - df['R_Knee_26'].dropna().values
        Lheel = 450 - df['L_Heel_29'].dropna().values
        Rheel = 450 - df['R_Heel_30'].dropna().values
        Ls = 450 - df['L_Shoulder_11'].dropna().values
        Rs = 450 - df['R_Shoulder_12'].dropna().values

        # 計算差異和比率
        Ldiff1 = Lknee - Lheel
        Rdiff1 = Rknee - Rheel
        Ldiff2 = Ls - Lheel
        Rdiff2 = Rs - Rheel
        Lper = Ldiff1 / Ldiff2
        Rper = Rdiff1 / Rdiff2

        Ldiff3 = np.zeros_like(Lper)  # 0
        Rdiff3 = Rper - Lper

        # 計算統計數據
        frame_count = len(Lknee)
        i = frame_count
        j = np.sum(Lper > Rper)  # 假設當 Lper 大於 Rper 時為 "左側更高"

        above = j / i
        below = 1 - above

        results_file.write(f"Frame : {i}\n")
        results_file.write("============================\n\n")
        results_file.write(f"Above Left Percentage: {above:f}\n")
        results_file.write(f"Below Left Percentage: {below:f}\n")

        if above >= below:
            results_file.write(f"Difference: {above - below:f}\n")
            if above - below >= 0.6:
                results_file.write("Leftside hurts seriously!\n")
            elif above - below >= 0.2:
                results_file.write("Leftside hurts!\n")
            else:
                results_file.write("Rightside is higher in more frames!\n")
        else:
            results_file.write(f"Difference: {below - above:f}\n")
            if below - above >= 0.6:
                results_file.write("Rightside hurts seriously!\n")
            elif below - above >= 0.2:
                results_file.write("Rightside hurts!\n")
            else:
                results_file.write("Leftside is higher in more frames!\n")

        results_file.write("============================")

        # 繪製圖表
        plt.figure()
        x = np.linspace(1, i, i)
        plt.plot(x, Lknee, color='r', marker='o', linewidth=2, markersize=2, label='L Knee')
        plt.plot(x, Rknee, color='b', marker='o', linewidth=2, markersize=2, label='R Knee')
        plt.title('Knee Original Data')
        plt.legend()
        #plt.savefig(os.path.join(output_folder, "Knee_Original_Data.png"))
        plt.close()

        plt.figure()
        plt.plot(x, Ldiff1, color='r', marker='o', linewidth=2, markersize=2, label='L Knee to Heel')
        plt.plot(x, Rdiff1, color='b', marker='o', linewidth=2, markersize=2, label='R Knee to Heel')
        plt.plot(x, Ldiff2, color='r', linestyle='--', linewidth=2, markersize=2, label='L Shoulder to Heel')
        plt.plot(x, Rdiff2, color='b', linestyle='--', linewidth=2, markersize=2, label='R Shoulder to Heel')
        plt.title('Body/Knee to Heel Range')
        plt.legend()
        #plt.savefig(os.path.join(output_folder, "Body_KneeToHeel_Range.png"))
        plt.close()

        plt.figure()
        plt.plot(x, Lper, color='r', marker='o', linewidth=2, markersize=2, label='L Percentage')
        plt.plot(x, Rper, color='b', marker='o', linewidth=2, markersize=2, label='R Percentage')
        plt.title('Percentage')
        plt.legend()
        #plt.savefig(os.path.join(output_folder, "Percentage.png"))
        plt.close()

        plt.figure()
        plt.plot(x, Ldiff3, color='r', marker='o', linewidth=2, markersize=2, label='L Diff3')
        plt.plot(x, Rdiff3, color='b', marker='o', linewidth=2, markersize=2, label='R Diff3')
        plt.title('Minus Trend')
        plt.legend()
        plt.savefig(os.path.join(output_folder, "Minus_Trend.png"))
        plt.close()