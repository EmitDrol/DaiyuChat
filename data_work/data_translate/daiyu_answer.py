import sys
import time

import jsonlines
sys.path.append('./')


from sys_prompts.DaiyuLin_sp import prompt
from model_work.openai_model import BaseOpenaiModel
from data_work.data_translate.utils.jsonl_processor import process_jsonl

from model_work.api_cfg import DEEPSEEK_API_KEY,DEEPSEEK_BASE_URL,DeepseekModels

def daiyu_answer(item, max_retries=8):
    """
    生成场景消息的函数，添加了重试机制。
    
    参数:
        item (dict): 包含场景信息的字典。
        max_retries (int): 最大重试次数，默认为3次。
        
    返回:
        dict: 更新后的item，包含生成的场景消息。
    """
    question = item['question']

    with jsonlines.open('data_work/data/human_quetions_processed_by_daiyu_answer.jsonl', 'r') as file:
        exisiting_questions = [i["scene_messages"][0]['content'] for i in  list(file)]
    
    if question in exisiting_questions:
        return None
    

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
            res_item = {}
            # 调用模型生成响应
            resp = model.call(question, max_tokens=8192,temperature=0.7,timeout=600,sys_prompt=prompt)
            res_item['scene_messages'] = [{
                "role": "user",
                "content": question
            }]
            res_item['scene_messages'].append({
                "role": "assistant",
                "content": resp
            })

            return res_item

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
    path = r'/cpfs01/shared/llm_ddd/chenyongkang/DaiyuChat/data_work/data/human_quetions.jsonl'
    process_jsonl(path,daiyu_answer,max_workers=8,add_mode=True)