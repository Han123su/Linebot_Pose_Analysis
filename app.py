import os
from process_video import process
from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *

app = Flask(__name__)

# Channel Access Token
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
# Channel Secret
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))


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
    result, phase_diff_images, lift_ratio_images, phase_diff_text, lift_ratio_text = process(video_path)

    # 回覆處理結果給使用者
    reply_messages = [TextSendMessage(text="影片處理完成，請查看以下結果。")]

    if phase_diff_text:
        reply_messages.append(TextSendMessage(text="Phase_diff.py 結果:\n" + phase_diff_text))

    if phase_diff_images:
        reply_messages.extend([ImageSendMessage(original_content_url=image_url, preview_image_url=image_url) for image_url in phase_diff_images])

    if lift_ratio_text:
        reply_messages.append(TextSendMessage(text="Lift_ratio.py 結果:\n" + lift_ratio_text))

    if lift_ratio_images:
        reply_messages.extend([ImageSendMessage(original_content_url=image_url, preview_image_url=image_url) for image_url in lift_ratio_images])

    line_bot_api.reply_message(event.reply_token, reply_messages)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
