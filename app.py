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

#app = Flask(__name__)
app = Flask(__name__, static_folder='static', static_url_path='/static')

# # Channel Access Token
# line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
line_bot_api = LineBotApi('ZKFSSh5O1UScyOoIOVZHPuSSQISeQgjzZIanIPQADT8iKXPzhUHn+0IPcUklijOKeChIcYemYwnrvzorDZ/J5nhQCSJxJ1Y5s0keI2sTBxuV8dO6T9Qs4w8ye0B5rNLR5VXlyziOYLWRvP40ZCg2UgdB04t89/1O/w1cDnyilFU=')
# # Channel Secret
# handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))
handler = WebhookHandler('ebddbbcefa93f0e69889881adf816763')


# 監聽所有來自 /callback 的 Post Request
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# 處理文字訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message = TextSendMessage(text="你是不是說: " + event.message.text)
    line_bot_api.reply_message(event.reply_token, message)

# 處理影片訊息
@handler.add(MessageEvent, message=VideoMessage)
def handle_video_message(event): 
    os.makedirs('static', exist_ok=True)
    message_content = line_bot_api.get_message_content(event.message.id)
    video_path = os.path.join("static", f"{event.message.id}.mp4")
    
    # 將影片儲存到本地
    with open(video_path, 'wb') as fd:
        for chunk in message_content.iter_content():
            fd.write(chunk)
    
    # 呼叫後端處理函式
    phase_diff_images, lift_ratio_images, phase_diff_text, lift_ratio_text = process(video_path)

    # 準備回覆消息
    reply_messages = []
    if phase_diff_text:
        reply_messages.append(TextSendMessage(text="*** 相位差分析結果 ***\n" + phase_diff_text))

    if phase_diff_images:
        reply_messages.extend([ImageSendMessage(original_content_url=image_url, preview_image_url=image_url) for image_url in phase_diff_images])

    if lift_ratio_text:
        reply_messages.append(TextSendMessage(text="*** 抬升高度比例分析結果 ***\n" + lift_ratio_text))

    if lift_ratio_images:
        reply_messages.extend([ImageSendMessage(original_content_url=image_url, preview_image_url=image_url) for image_url in lift_ratio_images])

    # 如果回覆消息數量超過 5 條，分批發送
    while len(reply_messages) > 0:
        chunk = reply_messages[:5]
        try:
            line_bot_api.reply_message(event.reply_token, chunk)
        except LineBotApiError as e:
            app.logger.error(f"LineBotApiError: {e}")
            # 確保在處理錯誤時不會重複發送
            break
        reply_messages = reply_messages[5:]

    # 在完成回覆後刪除 static 資料夾
    # static_folder = Path('static')
    # if static_folder.exists():
    #     shutil.rmtree(static_folder)
    #     app.logger.info("static 資料夾已刪除")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
