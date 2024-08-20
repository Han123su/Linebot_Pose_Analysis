import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 讀取Excel資料
PATH = "/data/2TSSD/Han212/skeleton/data/1016/1.25KG/"
data_file = "EachFrame.xlsx"
xlsx_file_path = os.path.join(PATH, data_file)

df = pd.read_excel(xlsx_file_path, engine='openpyxl')

# 檢查數據
print(df.head())

# 計算每個特徵的值
# Lknee = df['L_Knee_25'].dropna().values
# Rknee = df['R_Knee_26'].dropna().values
# Lheel = df['L_Heel_29'].dropna().values
# Rheel = df['R_Heel_30'].dropna().values
# Ls = df['L_Shoulder_11'].dropna().values
# Rs = df['R_Shoulder_12'].dropna().values

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

print(f"Frame Number: {i}")
print(f"Above Left Percentage: {above:f}")
print(f"Below Left Percentage: {below:f}")

if above >= below:
    print(f"Difference: {above - below:f}")
    if above - below >= 0.6:
        print("Leftside hurts seriously!")
    elif above - below >= 0.2:
        print("Leftside hurts!")
    else:
        print("Rightside is higher in more frames!")
else:
    print(f"Difference: {below - above:f}")
    if below - above >= 0.6:
        print("Rightside hurts seriously!")
    elif below - above >= 0.2:
        print("Rightside hurts!")
    else:
        print("Leftside is higher in more frames!")

# 創建儲存圖像的資料夾
image_folder = os.path.join(PATH, "image2")
os.makedirs(image_folder, exist_ok=True)

# 繪製圖表
plt.figure()
x = np.linspace(1,i,i)
plt.plot()
plt.plot(x, Lknee, color='r', marker='o', linewidth=2, markersize=2, label='L Knee')
plt.plot(x, Rknee, color='b', marker='o', linewidth=2, markersize=2, label='R Knee')
plt.title('Knee Original Data')
plt.legend()
plt.savefig(os.path.join(image_folder, "Knee_Original_Data.png"))

plt.figure()
plt.plot()
plt.plot(x, Ldiff1, color='r', marker='o', linewidth=2, markersize=2)
plt.plot(x, Rdiff1, color='b', marker='o', linewidth=2, markersize=2)
plt.plot(x, Ldiff2, color='r', marker='o', linewidth=2, markersize=2)
plt.plot(x, Rdiff2, color='b', marker='o', linewidth=2, markersize=2)
plt.title('Body/KneeToHeel Range')
#plt.legend()
plt.savefig(os.path.join(image_folder, "Body_KneeToHeel_Range.png"))

plt.figure()
plt.plot()
plt.plot(x, Lper, color='r', marker='o', linewidth=2, markersize=2)
plt.plot(x, Rper, color='b', marker='o', linewidth=2, markersize=2)
plt.title('Percentage')
#plt.legend()
plt.savefig(os.path.join(image_folder, "Percentage.png"))

plt.figure()
plt.plot()
plt.plot(x, Ldiff3, color='r', marker='o', linewidth=2, markersize=2)
plt.plot(x, Rdiff3, color='b', marker='o', linewidth=2, markersize=2)
plt.title('Minus Trend')
#plt.legend()
plt.savefig(os.path.join(image_folder, "Minus_Trend.png"))

plt.tight_layout()
plt.show()

print("Analysis and plotting completed!!!")
