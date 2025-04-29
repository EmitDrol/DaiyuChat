import sys
sys.path.append('./')

from data_work.data_translate.utils.jsonl_processor import process_jsonl
from data_work.data_translate.messages_to_trainningmsgs import replace_special_tags

def final_postprogress(item):
    messages = item['scene_messages']
    
    # 确保每条message包含role和content字段
    if not all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in messages):
        return None
    
    # 确保最后一条是assistant role
    while messages[-1]['role'] != 'assistant':
        messages.pop()
    
    
    # 确保user role下一条是 assistant role
    for i in range(len(messages)-1):
        if  messages[i]['role'] == 'user':
            if not messages[i+1]['role'] == 'assistant':
                return None
        if messages[i]['role'] == 'assistant':
            if not messages[i-1]['role'] == 'user' and not messages[i-1]['role'] == 'system':
                return None
        
    
    return item
if __name__ == '__main__':
    path = r'data_work/data/human_quetions_processed_by_daiyu_answer_processed_by_continue_talk_processed_by_c_talk_postprocess.jsonl'
    process_jsonl(path,final_postprogress,max_workers=1)