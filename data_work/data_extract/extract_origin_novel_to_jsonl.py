import re
import json


def extract_novel(script_file, output_file):
    # 读取整个剧本文件
    with open(script_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 正则表达式
    pattern = r"\n(?:第(?:\d+|[一二三四五六七八九十百千万]+)回)"

    # 使用 re.split 分割
    chapters = re.split(pattern, content)[1:]
    chapter_list = []
    for i,chapt in enumerate(chapters):
        # 构造JSON对象
        chapter_data = {
            "chapter_number": i+1,
            "chapter_title": chapt.split('\n')[0].strip(),
            "content": ''.join(chapt.split('\n')[1:]).strip()
        }
        chapter_list.append(chapter_data)

    # 写入JSONL文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for chapt in chapter_list:
            f.write(json.dumps(chapt, ensure_ascii=False) + '\n')


# 示例调用
script_file = 'data_work/data/红楼梦.txt'  
output_file = 'data_work/data/红楼梦.jsonl'  # 输出的JSONL文件路径

extract_novel(script_file, output_file)