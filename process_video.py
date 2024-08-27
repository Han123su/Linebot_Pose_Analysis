import os
import shutil
import time 
#from datetime import datetime
from flask import Flask, url_for 
from linebot.models import *
from Phase_diff_calculate import Phase_diff
from Lift_ratio_calculate import Lift_ratio
from Pose_tracking import pose_detect

app = Flask(__name__)

def process(video_path):

    pose_detect(video_path)

    # 檢查 Excel 檔案是否成功生成
    excel_file = "static/EachFrame.xlsx"
    if os.path.exists(excel_file):
        print("成功生成xlsx檔!!")
        # reply_message = TextSendMessage(text="成功生成xlsx檔!!")
        # return "Processing Complete", [reply_message]
    else:
        print("生成xlsx檔失敗，請檢查處理過程。")

    # 呼叫 Phase_diff 和 Lift_ratio 函數
    Phase_diff(excel_file)
    Lift_ratio(excel_file)

    phase_diff_image_folder = os.path.join('static', 'image')
    lift_ratio_image_folder = os.path.join('static', 'image2')

    phase_diff_images_urls = [url_for('static', filename=f'image/{img}', _external=True) for img in os.listdir(phase_diff_image_folder)]
    lift_ratio_images_urls = [url_for('static', filename=f'image2/{img}', _external=True) for img in os.listdir(lift_ratio_image_folder)]
    

    # timestamp = datetime.now().timestamp()  # 取得當前時間戳

    # phase_diff_images_urls = [
    #     url_for('static', filename=f'image/{img}', _external=True) + f"?v={timestamp}" 
    #     for img in os.listdir(phase_diff_image_folder)
    # ]
    # lift_ratio_images_urls = [
    #     url_for('static', filename=f'image2/{img}', _external=True) + f"?v={timestamp}"
    #     for img in os.listdir(lift_ratio_image_folder)
    # ]

    print("Phase Difference Images URLs:")
    for url in phase_diff_images_urls:
        print(url)
    
    print("Lift Ratio Images URLs:")
    for url in lift_ratio_images_urls:
        print(url)

    # 讀取 Phase_diff 和 Lift_ratio 生成的文字結果
    phase_diff_text_file = "static/phase_diff_results.txt"
    lift_ratio_text_file = "static/lift_ratio_results.txt"

    phase_diff_text = ""
    lift_ratio_text = ""

    try:
        with open(phase_diff_text_file, 'r') as file:
            phase_diff_text = file.read()
    except IOError as e:
        print(f"讀取 {phase_diff_text_file} 失敗: {e}")
        phase_diff_text = "讀取 Phase_diff 結果失敗。"

    try:
        with open(lift_ratio_text_file, 'r') as file:
            lift_ratio_text = file.read()
    except IOError as e:
        print(f"讀取 {lift_ratio_text_file} 失敗: {e}")
        lift_ratio_text = "讀取 Lift_ratio 結果失敗。"

    return phase_diff_images_urls, lift_ratio_images_urls, phase_diff_text, lift_ratio_text


def clear_static_folder():
    static_folder = 'static'
    if not os.path.exists(static_folder):
        return
    
    # 清除 static 資料夾中的所有文件和資料夾
    for filename in os.listdir(static_folder):
        file_path = os.path.join(static_folder, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                app.logger.info(f"Deleted file {file_path}")
            except Exception as e:
                app.logger.error(f"Failed to delete file {file_path}. Reason: {e}")
        elif os.path.isdir(file_path):
            try:
                # 刪除資料夾及其內容
                shutil.rmtree(file_path)
                app.logger.info(f"Deleted directory {file_path}")
            except Exception as e:
                app.logger.error(f"Failed to delete directory {file_path}. Reason: {e}")
    
    # 確保 `static/image` 和 `static/image2` 資料夾中的內容也被刪除
    for subfolder in ['image', 'image2']:
        folder_path = os.path.join(static_folder, subfolder)
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                app.logger.info(f"Deleted folder {folder_path}")
            except Exception as e:
                app.logger.error(f"Failed to delete folder {folder_path}. Reason: {e}")

    # 强制刷新文件系统缓存
    os.sync()

