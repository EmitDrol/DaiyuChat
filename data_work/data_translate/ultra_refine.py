import json
import time
import sys
sys.path.append('./')
from model_work.openai_model import BaseOpenaiModel
from model_work.prompts.ultra_refine import prompt_template,prompt_template_extracor
from data_work.data_translate.utils.jsonl_processor import process_jsonl
def ultra_refine(item, max_retries=8):
    """
    生成场景消息的函数，添加了重试机制。
    
    参数:
        item (dict): 包含场景信息的字典。
        max_retries (int): 最大重试次数，默认为3次。
        
    返回:
        dict: 更新后的item，包含生成的场景消息。
    """
    # scene_location = item['scene_location']
    scene_content = item['scene_content']
    scene_messges = item['scene_messages']

    # 初始化模型
    BASE_URL = "http://10.130.5.11:8000/v1"
    API_KEY = "empty"
    model_name = "R1_32B"
    model = BaseOpenaiModel(base_url=BASE_URL, api_key=API_KEY, model_name=model_name)

    retries = 0
    while retries <= max_retries:
        try:
            # 调用模型生成响应
            resp = model.call(prompt_template.format(json.dumps(scene_content,ensure_ascii=False)), temperature=0.7, timeout=600)
            new_scene_messges = prompt_template_extracor(resp)

            item['scene_messages'] = new_scene_messges
            return item

        except Exception as e:
            retries += 1
            sleep_time = 2 ** retries
            time.sleep(sleep_time)
            if retries > max_retries:
                print(f"调用模型失败，已达到最大重试次数({max_retries})。错误信息: {str(e)}")
            print(f"调用模型失败，正在进行第 {retries} 次重试。错误信息: {str(e)}")

if __name__ == '__main__':
    process_jsonl('/cpfs01/shared/llm_ddd/chenyongkang/DaiyuChat/data_work/data/87版红楼梦剧本_processed_by_scene_to_message_with_r1_32b_processed_by_msgs2trainingmsgs.jsonl',ultra_refine,max_workers=72,hard_mode=True)