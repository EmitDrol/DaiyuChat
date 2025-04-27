import json
import time
import sys
sys.path.append('./')
from model_work.api_cfg import DEEPSEEK_BASE_URL,DEEPSEEK_API_KEY,DeepseekModels,ALI_API_KEY,ALI_BASE_URL,AliModels
from model_work.openai_model import BaseOpenaiModel
from model_work.prompts.ultra_refine import format_prompt,prompt_template_extracor
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
    scene_messages = item['scene_messages']
    # 初始化模型
    BASE_URL = ALI_BASE_URL
    API_KEY = ALI_API_KEY
    model_name = AliModels.qwen_max.MODE_NAME
    model = BaseOpenaiModel(base_url=BASE_URL, api_key=API_KEY, model_name=model_name)

    while scene_messages[-1]['role'] != 'assistant':
        scene_messages.pop()
    
    scene_messges_backup = scene_messages.copy()
    
    retries = 0
    
    scene_messages = scene_messges_backup.copy()
    for i in range(len(scene_messages)):
        if scene_messages[i]['role'] == 'system':
            continue
        before_messages = scene_messages[:i+1] if i > 0 else []
        after_messages = scene_messages[i+1:] if i < len(scene_messages)-1 else []
        curent_message = scene_messages[i]
        prompt = format_prompt(before_messages, after_messages, curent_message)   
        while retries <= max_retries:
            try:
                print('=' * 50)
                resp = model.call(prompt, temperature=0.7, timeout=600)
                print(resp)
                break
            except Exception as e:
                retries += 1
                sleep_time = 2 ** retries
                time.sleep(sleep_time)
                if retries > max_retries:
                    print(f"调用模型失败，已达到最大重试次数({max_retries})。错误信息: {str(e)}")
                    break
                print(f"调用模型失败，正在进行第 {retries} 次重试。错误信息: {str(e)}")
                import traceback
                traceback.print_exc()

        new_curent_message = prompt_template_extracor(resp)
        scene_messages[i] = new_curent_message
    item['scene_messages'] = scene_messages
    return item
if __name__ == '__main__':
    process_jsonl('data_work/data/87版红楼梦剧本_processed_by_scene_to_message_dsv3_processed_by_msgs2trainingmsgs_processed_by_gen_background.jsonl',ultra_refine,max_workers=8,add_mode=True)