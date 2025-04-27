import json
import os
import jsonlines
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import Semaphore

def process_jsonl(path, func, max_workers=None, add_mode=False,epochs=1):
    """
    主函数，读取 JSONL 文件并使用信号量控制并发处理每个条目。
    每个条目生成后立即写入文件。
    
    参数：
    - path: 输入的 JSONL 文件路径。
    - func: 用于处理每个条目的函数。
    - max_workers: 最大并发线程数，默认为 CPU 核心数。
    - hard_mode: 是否覆盖已存在的输出文件。
    """
    # 读取输入文件
    with jsonlines.open(path, 'r') as file:
        items = list(file)

    items = items * epochs
    
    # 确定输出文件路径
    output_path = path.replace('.jsonl', f'_processed_by_{func.__name__}.jsonl')
    if os.path.exists(output_path) and not add_mode:
        raise FileExistsError(f"Output file {output_path} already exists.")

    # 设置最大并发数
    max_workers = max_workers if max_workers is not None else os.cpu_count()

    # 使用信号量限制并发任务数量
    semaphore = Semaphore(max_workers)

    def process_item(item):
        """封装处理函数，添加信号量控制"""
        with semaphore:  # 获取信号量许可
            return func(item)

    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务到线程池
        future_to_item = {executor.submit(process_item, item): item for item in items}

        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(future_to_item), total=len(items), desc='Processing'):
            try:
                processed_item = future.result()  # 阻塞直到任务完成
                if processed_item is not None:
                    # 打开输出文件
                    with jsonlines.open(output_path, 'a') as outfile:
                        outfile.write(processed_item)  # 写入文件
            except Exception as e:
                # print(f"Error processing item: {e}")
                import traceback
                traceback.print_exc()
                continue

    return output_path


def concat_jsonl_to_json(paths, output_path):
    """
    将多个 JSONL 文件合并为一个json 对象列表 json文件。
    
    参数：
    - paths: 输入的 JSONL 文件路径列表。
    - output_path: 输出的 JSON 文件路径。
    """
    merged_data = []
    for path in paths:
        with jsonlines.open(path, 'r') as file:
            merged_data.extend(list(file))
    with open(output_path, 'w') as file:
        json.dump(merged_data, file, ensure_ascii=False, indent=4)
        print(f"Merged data saved to {output_path}")

def concat_jsonl_to_jsonl(paths, output_path):
    """
    将多个 JSONL 文件合并为一个 JSONL 文件。
    
    参数：
    - paths: 输入的 JSONL 文件路径列表。
    - output_path: 输出的 JSONL 文件路径。
    """
    merged_data = []
    for path in paths:
        with jsonlines.open(path, 'r') as file:
            merged_data.extend(list(file))
    with jsonlines.open(output_path, 'w') as file:
        for item in merged_data:
            file.write(item)
        print(f"Merged data saved to {output_path}")

if __name__ == "__main__":
    inpaths = [
        "/cpfs01/shared/llm_ddd/chenyongkang/DaiyuChat/data_work/data/human_quetions_processed_by_daiyu_answer_processed_by_continue_talk_processed_by_c_talk_postprocess.jsonl",
        "/cpfs01/shared/llm_ddd/chenyongkang/DaiyuChat/data_work/data/87版红楼梦剧本_processed_by_scene_to_message_dsv3_processed_by_msgs2trainingmsgs_processed_by_gen_background_processed_by_ultra_refine.jsonl"
    ]
    concat_jsonl_to_json(inpaths, "data_work/data/train_data/train_data_v3.json")