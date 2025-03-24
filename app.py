
import os
from process_video import process
from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError, LineBotApiError
)
from linebot.models import *
from pathlib import Path
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
    if user_choices.get(user_id) != "back" or user_states.get(user_id) != "ready":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請先選擇『背面影片』才能上傳影片"))
        return

    user_states[user_id] = "processing"

    os.makedirs('static', exist_ok=True)
    message_content = line_bot_api.get_message_content(event.message.id)
    video_path = os.path.join("static", f"{event.message.id}.mp4")
    
    with open(video_path, 'wb') as fd:
        for chunk in message_content.iter_content():
            fd.write(chunk)
    
    phase_diff_images, lift_ratio_images, phase_diff_text, lift_ratio_text = process(video_path)

    reply_messages = []
    if phase_diff_text:
        reply_messages.append(TextSendMessage(text="*** 相位差分析結果 ***\n" + phase_diff_text))
    if phase_diff_images:
        reply_messages.extend([ImageSendMessage(original_content_url=url, preview_image_url=url) for url in phase_diff_images])
    if lift_ratio_text:
        reply_messages.append(TextSendMessage(text="*** 抬升高度比例分析結果 ***\n" + lift_ratio_text))
    if lift_ratio_images:
        reply_messages.extend([ImageSendMessage(original_content_url=url, preview_image_url=url) for url in lift_ratio_images])

    while reply_messages:
        chunk = reply_messages[:5]
        try:
            line_bot_api.reply_message(event.reply_token, chunk)
        except LineBotApiError as e:
            app.logger.error(f"LineBotApiError: {e}")
            break
        reply_messages = reply_messages[5:]

    # 清理 static 資料夾
    static_folder = Path('static')
    if static_folder.exists():
        for item in static_folder.iterdir():
            if item.is_dir() and item.name not in ['Image', 'Image2']:
                shutil.rmtree(item)
            elif item.is_file():
                item.unlink()

    # 回到選單狀態
    send_video_type_selection(user_id, event.reply_token, "分析完成！請選擇下一支影片類型：")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
