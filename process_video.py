import os
import shutil
import random
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

    phase_diff_image_folder = os.path.join('static', 'Image')
    lift_ratio_image_folder = os.path.join('static', 'Image2')

     # 確保資料夾存在
    if not os.path.exists(phase_diff_image_folder):
        print(f"資料夾 {phase_diff_image_folder} 不存在")
        return

    if not os.path.exists(lift_ratio_image_folder):
        print(f"資料夾 {lift_ratio_image_folder} 不存在")
        return

    # 生成圖片 URL
    # phase_diff_images_urls = [
    #     url_for('static', filename=f'Image/{img}', _external=True, _scheme='https') 
    #     for img in os.listdir(phase_diff_image_folder)
    # ]
    # lift_ratio_images_urls = [
    #     url_for('static', filename=f'Image2/{img}', _external=True, _scheme='https') 
    #     for img in os.listdir(lift_ratio_image_folder)
    # ]

    # 生成隨機亂碼的函數
    def generate_random_code():
        return str(random.randint(1000000000, 9999999999)) 

    phase_diff_images_urls = [
        url_for('static', filename=f'Image/{img}', _external=True, _scheme='https') + f'?v={generate_random_code()}'
        for img in os.listdir(phase_diff_image_folder)
    ]

    lift_ratio_images_urls = [
        url_for('static', filename=f'Image2/{img}', _external=True, _scheme='https') + f'?v={generate_random_code()}'
        for img in os.listdir(lift_ratio_image_folder)
    ]

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

