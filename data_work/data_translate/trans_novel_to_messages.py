import sys
import time
sys.path.append('./')


from model_work.prompts.trans_novel_to_message import prompt_template,prompt_template_extracor
from model_work.openai_model import BaseOpenaiModel
from data_work.data_translate.utils.jsonl_processor import process_jsonl

from model_work.api_cfg import DEEPSEEK_API_KEY,DEEPSEEK_BASE_URL,DeepseekModels

def novel_to_message(item, max_retries=8):
    """
    生成场景消息的函数，添加了重试机制。
    
    参数:
        item (dict): 包含场景信息的字典。
        max_retries (int): 最大重试次数，默认为3次。
        
    返回:
        dict: 更新后的item，包含生成的场景消息。
    """
    content = item['content']
    scene_messges = []

    # 初始化模型
    # BASE_URL = "http://10.130.5.11:8000/v1/"


    
    # API_KEY = "empty"
    # model_name = "r1_32b"
    BASE_URL=DEEPSEEK_BASE_URL
    API_KEY=DEEPSEEK_API_KEY
    model_name = DeepseekModels.deepseekv3.MODEL_NAME
    model = BaseOpenaiModel(base_url=BASE_URL, api_key=API_KEY, model_name=model_name)

    retries = 0
    while retries <= max_retries:
        try:
            # 调用模型生成响应
            resp = model.call(prompt_template.format(content), max_tokens=8192,temperature=0.7,timeout=600)
            messages = prompt_template_extracor(resp)

            # 更新场景消息并返回结果
            scene_messges.extend(messages)
            item['scene_messages'] = scene_messges
            return item

        except Exception as e:
            retries += 1
            sleep_time = 2 ** retries
            time.sleep(sleep_time)
            if retries > max_retries:
                print(f"调用模型失败，已达到最大重试次数({max_retries})。错误信息: {str(e)}")
            print(f"调用模型失败，正在进行第 {retries} 次重试。错误信息: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    path = r'/cpfs01/shared/llm_ddd/chenyongkang/DaiyuChat/data_work/data/红楼梦小说原本.jsonl'
    process_jsonl(path,novel_to_message,max_workers=8)