import os
from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import InvalidSignatureError
from linebot.models import *
from pathlib import Path
import shutil
import uuid
import subprocess
import shutil

# Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')

# LINE Channel Token and Secret
line_bot_api = LineBotApi('ZKFSSh5O1UScyOoIOVZHPuSSQISeQgjzZIanIPQADT8iKXPzhUHn+0IPcUklijOKeChIcYemYwnrvzorDZ/J5nhQCSJxJ1Y5s0keI2sTBxuV8dO6T9Qs4w8ye0B5rNLR5VXlyziOYLWRvP40ZCg2UgdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('ebddbbcefa93f0e69889881adf816763')

# 使用者選擇與狀態記錄
user_choices = {}  # 使用者影片類型選擇
user_states = {}   # 使用者狀態：waiting, ready, processing

# 快速選單函式
def send_video_type_selection(user_id, reply_token, welcome_text="請選擇你要分析的影片類型："):
    user_choices[user_id] = None
    user_states[user_id] = 'waiting'
    message = TextSendMessage(
        text=welcome_text,
        quick_reply=QuickReply(
            items=[
                QuickReplyButton(action=MessageAction(label="背面影片", text="選擇背面影片")),
                QuickReplyButton(action=MessageAction(label="側面影片", text="選擇側面影片")),
            ]
        )
    )
    line_bot_api.reply_message(reply_token, message)

# LINE webhook callback
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True) 
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# 使用者加入聊天室時自動跳出選單
@handler.add(FollowEvent)
def handle_follow(event):
    user_id = event.source.user_id
    send_video_type_selection(user_id, event.reply_token, "歡迎使用姿勢分析系統！請選擇你要分析的影片類型：")

# 處理文字訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    msg = event.message.text.strip()

    if msg == "選擇背面影片":
        user_choices[user_id] = "back"
        user_states[user_id] = "ready"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="你已選擇背面影片，請上傳影片進行分析"))
    
    elif msg == "選擇側面影片":
        user_choices[user_id] = "side"
        user_states[user_id] = "waiting"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="⚠️目前僅支援背面影片，請重新選擇"))
        send_video_type_selection(user_id, event.reply_token)

    else:
        state = user_states.get(user_id, 'waiting')
        if state == "ready":
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請上傳背面影片（.mp4）進行分析"))
        else:
            send_video_type_selection(user_id, event.reply_token, "請先選擇你要分析的影片類型：")

# 處理影片訊息
@handler.add(MessageEvent, message=VideoMessage)
def handle_video_message(event):
    user_id = event.source.user_id
    user_choice = user_choices.get(user_id, None)

    if user_choice != "back":
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="⚠️請先選擇影片角度（背面）後再上傳影片。")
        )
        send_video_type_selection(user_id, event.reply_token)
        return

    # 1. 儲存影片
    message_id = event.message.id
    ext = '.mp4'
    video_tempfile_path = os.path.join('static', f"{uuid.uuid4().hex}{ext}")
    with open(video_tempfile_path, 'wb') as f:
        for chunk in line_bot_api.get_message_content(message_id).iter_content():
            f.write(chunk)

    # 2. 呼叫骨架分析 (Pose_tracking_back_withBall)
    xlsx_path = video_tempfile_path.replace('.mp4', '.xlsx')
    subprocess.run([
        "python", "Pose_tracking_back_withBall.py",
        "--video", video_tempfile_path,
        "--output", xlsx_path
    ])

    # 3. 呼叫分析主程式 (analyze_main.py)
    result = subprocess.run([
        "python", "analyze_main.py",
        "--input", xlsx_path,
        "--image_folder", "result_images"
    ], capture_output=True, text=True)

    # 4. 回傳結果給使用者
    if result.returncode == 0:
        reply_text = result.stdout[-5000:] if result.stdout else "分析完成，無輸出內容"
    else:
        reply_text = f"分析失敗：{result.stderr}"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

    # 5. 清除中繼資料夾
    for folder in ['static/FRAMES', 'static/FRAMES_MODIFY', 'static/FRAMES_TRACKING']:
        if os.path.exists(folder):
            shutil.rmtree(folder)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
