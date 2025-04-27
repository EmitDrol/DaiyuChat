import sys
sys.path.append('./')

from data_work.data_translate.utils.jsonl_processor import process_jsonl
from data_work.data_translate.messages_to_trainningmsgs import replace_special_tags

def c_talk_postprocess(item):
    messages = item['scene_messages']
    for message in messages:
        if message['role'] == 'user':
            message['content'] = replace_special_tags(message['content'])
    return item
if __name__ == '__main__':
    path = r'data_work/data/human_quetions_processed_by_daiyu_answer_processed_by_continue_talk.jsonl'
    process_jsonl(path,c_talk_postprocess,max_workers=8,add_mode=True)