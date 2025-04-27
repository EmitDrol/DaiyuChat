import sys
import time
import random
import jsonlines
sys.path.append('./')


from sys_prompts.DaiyuLin_sp import prompt as daiyu_sys_prompt
from model_work.prompts.continue_talk import sys_prompt

from model_work.openai_model import BaseOpenaiModel
from data_work.data_translate.utils.jsonl_processor import process_jsonl

from model_work.api_cfg import DEEPSEEK_API_KEY,DEEPSEEK_BASE_URL,DeepseekModels

def trans_message(message: dict):
    try:
        if 'role' in message:
            new_message = message.copy()  # 创建副本
            if new_message['role'] == 'user':
                new_message['role'] = 'assistant'
            elif new_message['role'] == 'assistant':
                new_message['role'] = 'user'
            return new_message
        else:
            raise ValueError("字典中不存在'role'字段")
    except Exception as e:
        print(f"wrong in trans_message，错误信息: {str(e)}")
        return None

def model_call_with_message(messages: list, max_retries=8, system_prompt=None):
    BASE_URL = DEEPSEEK_BASE_URL
    API_KEY = DEEPSEEK_API_KEY
    model_name = DeepseekModels.deepseekv3.MODEL_NAME
    model = BaseOpenaiModel(base_url=BASE_URL, api_key=API_KEY, model_name=model_name)
    
    retries = 0
    while retries <= max_retries:
        try:
            resp = model.messages_call(messages=messages[:], sys_prompt=system_prompt)
            print('=' * 50)
            print(resp)
            updated_messages = messages[:]  # 创建副本
            updated_messages.append({
                "role": "assistant",
                "content": resp
            })
            return updated_messages
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
    return messages  # 返回原始消息以防所有重试失败

def continue_talk(item):
    """
    生成场景消息的函数，添加了重试机制。
    
    参数:
        item (dict): 包含场景信息的字典。
        
    返回:
        dict: 更新后的item，包含生成的场景消息。
    """
    messages_daiyu_assis = item.get('scene_messages', [])
    
    user_sys_prompt = sys_prompt()
    continue_round = random.randint(1, 5)
    
    for _ in range(continue_round):
        # 转换消息角色
        messages_user_assis = []
        for message in messages_daiyu_assis:
            transformed_message = trans_message(message)
            if transformed_message is not None:
                messages_user_assis.append(transformed_message)
            else:
                print(f"消息转换失败: {message}")
                return item  # 提前返回以防错误传播
        
        # 调用模型生成用户侧对话
        messages_user_assis = model_call_with_message(messages_user_assis, system_prompt=user_sys_prompt)
        if messages_user_assis is None:
            print("模型调用失败，无法继续生成对话")
            return item
        
        # 再次转换消息角色
        messages_daiyu_assis = []
        for message in messages_user_assis:
            transformed_message = trans_message(message)
            if transformed_message is not None:
                messages_daiyu_assis.append(transformed_message)
            else:
                print(f"消息转换失败: {message}")
                return item
        
        # 调用模型生成黛玉侧对话
        messages_daiyu_assis = model_call_with_message(messages_daiyu_assis, system_prompt=daiyu_sys_prompt)
        if messages_daiyu_assis is None:
            print("模型调用失败，无法继续生成对话")
            return None
    
    item['scene_messages'] = messages_daiyu_assis
    return item


    


if __name__ == '__main__':
    path = r'data_work/data/human_quetions_processed_by_daiyu_answer.jsonl'
    process_jsonl(path,continue_talk,max_workers=8)