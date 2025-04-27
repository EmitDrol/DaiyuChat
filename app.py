from flask import Flask, Response, request, jsonify, render_template, send_from_directory
import openai
from openai import OpenAI
import os
import json
from datetime import datetime  # 引入时间模块

from myutils.fomat_time import formatted_time
from sys_prompts.DaiyuLin_sp import prompt as Daiyu_prompt, opening as Daiyu_openning

app = Flask(__name__)

sys_prompt = Daiyu_prompt
opening_message = Daiyu_openning

## =========================== ##
api_key = "empty"
base_url = "http://10.130.5.11:8000/v1"
model = 'daiyu_20250426_042114'

## =========================== ##

if base_url:
    openai.api_base = base_url  # 设置自定义 Base URL

# 定义对话保存目录和文件名格式
DIALOGUE_DIR = "dialogues"  # 对话文件保存目录
os.makedirs(DIALOGUE_DIR, exist_ok=True)  # 确保目录存在


def save_dialogue_to_jsonl(user_messages, model_reply):
    """
    将对话保存到 JSONL 文件中。
    每次对话生成一个新的文件，文件名包含时间戳。
    """
    timestamp = formatted_time()
    file_path = os.path.join(DIALOGUE_DIR, timestamp+'.json')

    dialogue_entry = {
        "timestamp": timestamp,
        "user_messages": user_messages,
        "model_reply": model_reply
    }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dialogue_entry,fp=f,ensure_ascii=False,indent=4)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/welcome', methods=['GET'])
def welcome():
    # 返回开场白内容
    return jsonify({
        "message": opening_message(),
    })


@app.route('/chat', methods=['POST'])
def chat():
    user_messages = request.json.get('messages')
    if not user_messages:
        return jsonify({"error": "Message is required"}), 400

    try:
        messages = [{"role": "system", "content": sys_prompt}]
        messages += user_messages

        # 初始化 OpenAI 客户端
        client = openai.OpenAI(base_url=base_url, api_key=api_key)

        # 使用流式调用 OpenAI GPT 模型
        response_generator = client.chat.completions.create(
            temperature=0.7,
            model=model,
            messages=messages,
            stream=True  # 启用流式传输
        )

        def generate():
            full_reply = ""
            for chunk in response_generator:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_reply += content
                    yield content  # 流式返回每个片段
            # 保存完整对话到 JSONL 文件
            save_dialogue_to_jsonl(user_messages, full_reply)

        # 返回流式响应
        return Response(generate(), content_type='text/plain')

    except Exception as e:
        print(e)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500



@app.route('/resource/<path:filename>')
def resource(filename):
    return send_from_directory('resource', filename)


if __name__ == '__main__':
    app.run(debug=True,port=56674)