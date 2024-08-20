import os
from linebot.models import *

def process(video_path):
    # 執行 Pose_tracking.py
    os.system(f"python Pose_tracking.py {video_path}")

    # 檢查 Excel 檔案是否成功生成
    excel_file = "static/EachFrame.xlsx"
    if os.path.exists(excel_file):
        print(text="成功生成xlsx檔!!")
    else:
        print("生成xlsx檔失敗，請檢查處理過程。")

    # # 執行 Phase_diff.py 和 Lift_ratio.py
    # os.system("python Phase_diff.py")
    # os.system("python Lift_ratio.py")

    # # 讀取生成的圖片
    # image_folder = "image"
    # image2_folder = "image2"
    # image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
    # image2_files = [os.path.join(image2_folder, img) for img in os.listdir(image2_folder)]

    # # 假設圖片可以從伺服器的 URL 存取
    # # 這裡需要提供你的圖片儲存伺服器的 URL 根路徑
    # base_url = "https://your-server-url.com/"
    # images = [base_url + img for img in image_files + image2_files]

    # return "Processing Complete", images

