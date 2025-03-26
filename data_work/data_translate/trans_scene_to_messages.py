import sys
sys.path.append('./')

import jsonlines
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from model_work.prompts.trans_scene_to_messages import prompt_template_v1 as scene_to_messages_prompt, \
    prompt_template_v1_extracor as scene_to_messages_extracor
from model_work.openai_model import BaseOpenaiModel


def gen_message(item):
    """
    生成场景消息的函数。
    """
    scene_location = item['scene_location']
    scene_content = item['scene_content']
    scene_messges = [{'role': 'Narrator', 'content': f'在{scene_location}'}]

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    API_KEY = "sk-60c35b95f17f42a0bd50452648b7e07e"
    # 初始化模型
    model = BaseOpenaiModel(base_url=BASE_URL, api_key=API_KEY, model_name="qwen2-72b-instruct")
    resp = model.call(scene_to_messages_prompt.format(scene_content), temperature=0.7)
    messages = scene_to_messages_extracor(resp)

    scene_messges.extend(messages)
    item['scene_messges'] = scene_messges
    return item


def main(path):
    """
    主函数，读取 JSONL 文件并并发处理每个条目。
    每个条目生成后立即写入文件。
    """
    with jsonlines.open(path, 'r') as file:
        items = list(file)

    # 打开输出文件，使用追加模式（append mode）
    output_path = path.replace('.jsonl', '_messages.jsonl')
    with jsonlines.open(output_path, 'a') as outfile:
        # 使用线程池并发处理
        max_workers = 16  # 设置最大并发数

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务到线程池
            future_to_item = {executor.submit(gen_message, item): item for item in items}

            # 使用 tqdm 显示进度条
            for future in tqdm(as_completed(future_to_item), total=len(items), desc='Processing'):
                try:
                    processed_item = future.result()
                    # 立即写入文件
                    outfile.write(processed_item)
                except Exception as e:
                    print(f"Error processing item: {e}")


if __name__ == '__main__':
    path = r'data_work\data\87版红楼梦剧本.jsonl'
    main(path)