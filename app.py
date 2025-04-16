from flask import Flask, request, jsonify, render_template, send_from_directory
import openai
from openai import OpenAI
import os
import json
from datetime import datetime  # 引入时间模块

from myutils.fomat_time import formatted_time
from sys_prompts.DaiyuLin import prompt as Daiyu_prompt, opening as Daiyu_openning

app = Flask(__name__)

# 配置 envs
ali_key = "sk-60c35b95f17f42a0bd50452648b7e07e"
ali_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ali_models = ["qwen2-72b-instruct", "qwen2-7b-instruct"]

deepseek_key = "sk-368172d0f1284782b87d8a1e5bfd4586"
deepseek_base_url = "https://api.deepseek.com"
deepseek_models = ["deepseek-chat"]

sys_prompt = Daiyu_prompt
opening_message = Daiyu_openning

## =========================== ##
api_key = ali_key
base_url = ali_base_url
model = ali_models[0]
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

        # 调用 OpenAI GPT 模型
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            temperature=0.7,
            model=model,
            messages=messages,
        )
        # 获取模型的回复
        model_reply = response.choices[0].message.content.strip()

        # 保存对话到 JSONL 文件
        save_dialogue_to_jsonl(user_messages, model_reply)

        return jsonify({"reply": model_reply})
    except Exception as e:
        print(e)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/resource/<path:filename>')
def resource(filename):
    return send_from_directory('resource', filename)


if __name__ == '__main__':
    app.run(debug=True)