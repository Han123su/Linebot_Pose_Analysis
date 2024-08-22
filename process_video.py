import os
import subprocess
from flask import url_for 
from linebot.models import *
from Phase_diff_calculate import Phase_diff
from Lift_ratio_calculate import Lift_ratio

def process(video_path):

    try:
        subprocess.run(['python', 'Pose_tracking.py', video_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

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

    # # 讀取 Phase_diff 和 Lift_ratio 生成的圖片
    # phase_diff_image_folder = os.path.join('static', 'image')
    # phase_diff_images = [os.path.join(phase_diff_image_folder, img) for img in os.listdir(phase_diff_image_folder)]
    
    # lift_ratio_image_folder = os.path.join('static', 'image2')
    # lift_ratio_images = [os.path.join(lift_ratio_image_folder, img) for img in os.listdir(lift_ratio_image_folder)]

    # # 假設圖片可以從伺服器的 URL 存取
    # base_url = os.getenv('BASE_URL', 'https://your-server-url.com/')
    
    # phase_diff_images_urls = [base_url + img for img in phase_diff_images]
    # lift_ratio_images_urls = [base_url + img for img in lift_ratio_images]

    phase_diff_image_folder = os.path.join('static', 'image')
    lift_ratio_image_folder = os.path.join('static', 'image2')

    phase_diff_images_urls = [url_for('static', filename=f'image/{img}', _external=True) for img in os.listdir(phase_diff_image_folder)]
    lift_ratio_images_urls = [url_for('static', filename=f'image2/{img}', _external=True) for img in os.listdir(lift_ratio_image_folder)]

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

    return "Processing Complete", phase_diff_images_urls, lift_ratio_images_urls, phase_diff_text, lift_ratio_text


def clear_static_folder():  # 清理 static 資料夾中的舊檔案
    static_folder = 'static'
    if not os.path.exists(static_folder):
        return  # 如果資料夾不存在，直接返回
    
    for filename in os.listdir(static_folder):
        file_path = os.path.join(static_folder, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                app.logger.info(f"Deleted {file_path}")
            except Exception as e:
                app.logger.error(f"Failed to delete {file_path}. Reason: {e}")