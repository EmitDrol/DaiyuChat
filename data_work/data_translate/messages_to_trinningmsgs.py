import sys
sys.path.append('./')
from data_work.data_translate.utils.jsonl_processor import process_jsonl

def remove_special_tags(content):
    """去除消息中的特殊标签<thinking>和<action>"""
    return content.replace("<thinking>", "").replace("</thinking>", "").replace("<action>", "").replace("</action>", "")

def msgs2trainingmsgs(data):
    # 确保输入数据包含scene_messages字段
    if "scene_messages" not in data:
        raise ValueError("输入字典缺少'scene_messages'字段")
    
    messages = data["scene_messages"]
    if not isinstance(messages, list):
        raise ValueError("字段'scene_messages'必须是一个列表")
    
    try:
        if "黛玉" not in [x["role"] for x in messages]:
            # print("输入字典中不存在'黛玉'角色的消息")
            return None
    except Exception as e:
        print(f"输入字典中不存在'黛玉'角色的消息，错误信息: {str(e)}")
        return None
    processed_messages = []

    # 遍历消息列表进行处理
    for i, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content", "")

        if i == 0:  # 第一条消息角色改为system
            processed_messages.append({"role": "system", "content": content})
        elif role == "黛玉":  # 黛玉的消息角色改为assistant
            # 保留特殊标签
            processed_messages.append({"role": "assistant", "content": content.strip()})
        else:  # 其余角色消息合并，并保留角色信息
            cleaned_content = remove_special_tags(content)
            annotated_content = f"[{role}]：{cleaned_content.strip()}"
            
            if processed_messages and processed_messages[-1]["role"] == "user":
                # 如果上一条消息是user，则合并内容
                processed_messages[-1]["content"] += "\n" + annotated_content
            else:
                # 否则新增一条user消息
                processed_messages.append({"role": "user", "content": annotated_content})

    # 更新字典中的scene_messages字段
    data["scene_messages"] = processed_messages
    return data

if __name__ == "__main__":
    process_jsonl("/cpfs01/shared/llm_ddd/chenyongkang/DaiyuChat/data_work/data/87版红楼梦剧本_processed_by_scene_to_message_with_r1_32b.jsonl", msgs2trainingmsgs, hard_mode=True)