import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import pandas as pd
import sys

# 讀取命令行參數中的影片路徑
if len(sys.argv) != 2:
    print("Usage: python Pose_tracking.py <video_file_path>")
    sys.exit(1)

video_file_path = sys.argv[1]

# 嘗試開啟影片
vidcap = cv2.VideoCapture(video_file_path)
if not vidcap.isOpened():
    print(f"Error: Could not open video file '{video_file_path}'.")
    sys.exit(1)

success, image = vidcap.read()
count = 0

# 設定路徑
base_path = os.path.dirname(video_file_path)
os.makedirs(os.path.join(base_path, 'FRAMES'), exist_ok=True)
os.makedirs(os.path.join(base_path, 'FRAMES_MP'), exist_ok=True)

# 將影片分割成幀，並存入指定文件夾
while success:
    cv2.imwrite(os.path.join(base_path, "FRAMES", f"{count}.jpg"), image)  # save frame as JPEG file      
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

vidcap.release()

# 初始化 pose_data
index = [
    'keypoint', 'L_Heel_29', 'R_Heel_30', 
    'L_Knee_25', 'R_Knee_26', 
    'L_Shoulder_11', 'R_Shoulder_12', 
    'L_Hip_23', 'R_Hip_24'
]
pose_data = []

# 抓取剛剛分割好的幀，並儲存到 frame_list 資料結構中
path = os.path.join(base_path, 'FRAMES', '*.jpg')
#result = os.path.join(base_path, 'output.mp4')
frame_list = sorted(glob.glob(path), key=os.path.getmtime)

print("frame count: ", len(frame_list))

fps = 30
shape = cv2.imread(frame_list[0]).shape  # delete dimension 3
size = (shape[1], shape[0])
print("frame size: ", size)
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter(result, fourcc, fps, size)

# 設定 MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

count = 0

# 開始用 MediaPipe 處理畫面
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    for idx, path in enumerate(frame_list):
        frame = cv2.imread(path)

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=4),
        )

        # 將 MediaPipe 結果存入 pose_data
        try:
            landmarks = results.pose_landmarks.landmark
            row = [
                None,  # keypoint 列保持為空
                landmarks[29].y * 450 if len(landmarks) > 29 else None,  # L_Heel_29
                landmarks[30].y * 450 if len(landmarks) > 30 else None,  # R_Heel_30
                landmarks[25].y * 450 if len(landmarks) > 25 else None,  # L_Knee_25
                landmarks[26].y * 450 if len(landmarks) > 26 else None,  # R_Knee_26
                landmarks[11].y * 450 if len(landmarks) > 11 else None,  # L_Shoulder_11
                landmarks[12].y * 450 if len(landmarks) > 12 else None,  # R_Shoulder_12
                landmarks[23].y * 450 if len(landmarks) > 23 else None,  # L_Hip_23
                landmarks[24].y * 450 if len(landmarks) > 24 else None   # R_Hip_24
            ]
            pose_data.append(row)
        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
            pose_data.append([None] * len(index))  # 若無法獲取數據，則填充空值

        current_frame = idx + 1
        total_frame_count = len(frame_list)
        percentage = int(current_frame * 30 / (total_frame_count + 1))
        print("\rProcess: [{}{}] {:06d} / {:06d}".format("#" * percentage, "." * (30 - 1 - percentage), current_frame, total_frame_count), end='')

        cv2.imwrite(os.path.join(base_path, "FRAMES_MP", f"{count}.jpg"), frame)
        count += 1
        #out.write(frame)

#out.release()
print("\nPose Tracking and Data Export Completed !!!")

# 將數據直接轉為 XLSX
df = pd.DataFrame(pose_data, columns=index)
xlsx_file_path = os.path.join(base_path, 'EachFrame.xlsx')
df.to_excel(xlsx_file_path, index=False, engine='openpyxl')

print(f"Data has been exported to XLSX file '{xlsx_file_path}'.")
