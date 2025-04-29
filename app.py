from flask import Flask, Response, request, jsonify, render_template, send_from_directory
import openai
from openai import OpenAI
import os
import json
from datetime import datetime
import threading  # 引入线程支持

from myutils.fomat_time import formatted_time
from sys_prompts.DaiyuLin_sp import prompt as Daiyu_prompt, opening as Daiyu_openning
from model_work.api_cfg import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DeepseekModels
from rag.rag import load_and_index_documents, retrieve
# 添加全局变量用于控制 RAG 的启用状态
enable_rag = True  # 默认启用 RAG

app = Flask(__name__)

sys_prompt = Daiyu_prompt
opening_message = Daiyu_openning

## =========================== ##
# api_key = DEEPSEEK_API_KEY
# base_url = DEEPSEEK_BASE_URL
# model = DeepseekModels.deepseekv3.MODEL_NAME
api_key = 'api_key'
base_url = 'http://10.130.5.11:8000/v1'
model = 'daiyu_20250427_162043'

## =========================== ##

# 定义对话保存目录和文件名格式
DIALOGUE_DIR = "dialogues"  # 对话文件保存目录
os.makedirs(DIALOGUE_DIR, exist_ok=True)  # 确保目录存在

# 全局变量用于存储知识库索引
knowledge_base_index = None
kb_loading_lock = threading.Lock()  # 线程锁，确保知识库加载是线程安全的
kb_loaded = False  # 标志位，表示知识库是否已加载

# 加载知识库索引
def load_knowledge_base():
    """加载知识库索引"""
    global knowledge_base_index, kb_loaded
    print("Loading knowledge base...")
    try:
        knowledge_base_index = load_and_index_documents()
        print("Knowledge base loaded successfully.")
    except Exception as e:
        print(f"Failed to load knowledge base: {e}")
    finally:
        kb_loaded = True


# 异步加载知识库
def async_load_kb():
    global knowledge_base_index, kb_loaded
    with kb_loading_lock:  # 确保只有一个线程执行加载操作
        if not kb_loaded:  # 如果知识库尚未加载，则加载
            load_knowledge_base()


# 启动时异步加载知识库
threading.Thread(target=async_load_kb, daemon=True).start()

# 检索并提示增强
def retrieval_augmented(input,query) -> str:
    global knowledge_base_index, kb_loaded
    try:
        if knowledge_base_index is None:
            print("Knowledge base not loaded yet.")
            return input  # 如果知识库未加载完成，直接返回原始查询

        retrieved_texts = retrieve(knowledge_base_index, query, top_k=3)
        template = f"\n<story>{''.join([f'[[{i}]]. {result}' for i, result in retrieved_texts])}<story>"
        return input + template
    except Exception as e:
        import traceback
        traceback.print_exc()
        return input


def save_dialogue_to_jsonl(user_messages, model_reply):
    """
    将对话保存到 JSONL 文件中。
    每次对话生成一个新的文件，文件名包含时间戳。
    """
    timestamp = formatted_time()
    file_path = os.path.join(DIALOGUE_DIR, timestamp + '.json')

    dialogue_entry = {
        "timestamp": timestamp,
        "user_messages": user_messages,
        "model_reply": model_reply
    }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dialogue_entry, fp=f, ensure_ascii=False, indent=4)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/welcome', methods=['GET'])
def welcome():
    # 返回开场白内容
    return jsonify({
        "message": opening_message(),
    })

@app.route('/toggle-rag', methods=['POST'])
def toggle_rag():
    """
    切换 RAG 的启用状态。
    用户可以通过 POST 请求动态开启或关闭 RAG。
    """
    global enable_rag
    data = request.json
    enable_rag = data.get('enable', True)  # 默认值为 True
    return jsonify({"status": "success", "enable_rag": enable_rag})


@app.route('/chat', methods=['POST'])
def chat():
    user_messages = request.json.get('messages')
    if not user_messages:
        return jsonify({"error": "Message is required"}), 400

    try:
        # 检查是否启用了 RAG，并根据需要增强用户输入
        if enable_rag:
            user_messages[-1] = {
                "role": "user",
                "content": retrieval_augmented(input=user_messages[-1]['content'], query=user_messages[-1]['content'])
            }

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


@app.route('/get-rag-status', methods=['GET'])
def get_rag_status():
    """
    获取当前 RAG 的启用状态。
    """
    return jsonify({"enable_rag": enable_rag})


@app.route('/resource/<path:filename>')
def resource(filename):
    return send_from_directory('resource', filename)


if __name__ == '__main__':
    app.run(debug=False, port=9981)