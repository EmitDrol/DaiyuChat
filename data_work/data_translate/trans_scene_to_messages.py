import sys
import time
sys.path.append('./')


from model_work.prompts.trans_scene_to_messages import prompt_template_v1 as scene_to_messages_prompt, \
    prompt_template_v1_extracor as scene_to_messages_extracor
from model_work.openai_model import BaseOpenaiModel
from data_work.data_translate.utils.jsonl_processor import process_jsonl


def scene_to_message(item, max_retries=8):
    """
    生成场景消息的函数，添加了重试机制。
    
    参数:
        item (dict): 包含场景信息的字典。
        max_retries (int): 最大重试次数，默认为3次。
        
    返回:
        dict: 更新后的item，包含生成的场景消息。
    """
    scene_location = item['scene_location']
    scene_content = item['scene_content']
    if '黛玉' not in scene_content:
        return None
    scene_messges = [{'role': 'Narrator', 'content': f'在{scene_location}'}]

    from model_work.api_cfg import DEEPSEEK_API_KEY,DEEPSEEK_BASE_URL,DeepseekModels
    # 初始化模型
    BASE_URL = DEEPSEEK_BASE_URL
    API_KEY = DEEPSEEK_API_KEY
    model_name = DeepseekModels.deepseekv3.MODEL_NAME
    model = BaseOpenaiModel(base_url=BASE_URL, api_key=API_KEY, model_name=model_name)

    retries = 0
    while retries <= max_retries:
        try:
            # 调用模型生成响应
            resp = model.call(scene_to_messages_prompt.format(scene_content), temperature=0.7, timeout=600)
            messages = scene_to_messages_extracor(resp)

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


if __name__ == '__main__':
    path = r'/cpfs01/shared/llm_ddd/chenyongkang/DaiyuChat/data_work/data/87版红楼梦剧本.jsonl'
    process_jsonl(path,scene_to_message,max_workers=8,add_mode=True,epochs=8)