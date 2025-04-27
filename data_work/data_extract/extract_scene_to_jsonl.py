import re
import json


def extract_scenes(script_file, titles, output_file):
    """
    提取剧本中的每一幕，并保存为JSONL格式。

    :param script_file: 剧本文件路径
    :param titles: 每集标题列表
    :param output_file: 输出的JSONL文件路径
    """
    # 读取整个剧本文件
    with open(script_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 初始化结果列表
    scenes_data = []

    # 按标题分割剧本内容
    for i in range(len(titles) - 1):
        current_title = titles[i]
        next_title = titles[i + 1]

        # 使用正则表达式提取当前集的内容
        pattern = re.escape(current_title) + r'(.*?)' + re.escape(next_title)
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            continue

        episode_content = match.group(1).strip()

        # 提取每一幕
        scene_pattern = r'(\d+)\s+(.*?)(?=\d+\s+|\Z)'
        scenes = re.findall(scene_pattern, episode_content, re.MULTILINE | re.DOTALL)

        for scene_number, scene_info in scenes:
            # 分离场景地点和内容
            location_match = re.match(r'(.*?)\n', scene_info, re.DOTALL)
            if location_match:
                scene_location = location_match.group(1).strip()
                scene_content = scene_info[len(scene_location):].strip()
            else:
                scene_location = ""
                scene_content = scene_info.strip()

            # 构造JSON对象
            scene_data = {
                "episode_number": i+1,
                "episode_title": current_title,
                "scene_number": int(scene_number),
                "scene_location": scene_location,
                "scene_content": scene_content
            }
            scenes_data.append(scene_data)

    # 写入JSONL文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for scene in scenes_data:
            f.write(json.dumps(scene, ensure_ascii=False) + '\n')


# 示例调用
script_file = 'data_work\data\87版红楼梦剧本.txt'  # 剧本文件路径
titles = ['序集', '第一集 黛玉进府', '第二集 金玉初识', '第三集 淫丧天香楼', '第四集 协理宁国府',
          '第五集 元妃省亲', '第六集 宝玉参禅', '第七集 叔嫂染恙', '第八集 黛玉葬花', '第九集 宝玉受答',
          '第十集 二进荣国府', '第十一集 凤姐泼醋', '第十二集 鸳鸯抗婚', '第十三集 踏雪寻梅',
          '第十四集 晴雯补裘', '第十五集 探春理家', '第十六集 怡红夜宴', '第十七集 三姐饮剑',
          '第十八集 二姐吞金', '第十九集 抄检大观园', '第二十集 宝玉探晴雯', '第二十一集 诸芳流散',
          '第二十二集 误窃通灵', '第二十三集 探春远嫁', '第二十四集 黛玉之死', '第二十五集 贾府抄没',
          '第二十六集 狱庙相逢', '第二十七集 悬崖撒手']
output_file = 'data_work\data\87版红楼梦剧本.jsonl'  # 输出的JSONL文件路径

extract_scenes(script_file, titles, output_file)