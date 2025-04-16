from concurrent.futures import ThreadPoolExecutor,as_completed
import os
import jsonlines
from tqdm import tqdm


def process_jsonl(path,func,max_workers=None,hard_mode=False):
    """
    主函数，读取 JSONL 文件并并发处理每个条目。
    每个条目生成后立即写入文件。
    """
    with jsonlines.open(path, 'r') as file:
        items = list(file)

    # 打开输出文件，使用追加模式（append mode）
    output_path = path.replace('.jsonl', f'_processed_by_{func.__name__}.jsonl')
    # print(f"Writing to {output_path}")
    if os.path.exists(output_path) and not hard_mode:
        raise FileExistsError(f"Output file {output_path} already exists.")
    with jsonlines.open(output_path, 'a') as outfile:
        # 使用线程池并发处理
        max_workers = max_workers if max_workers is not None else os.cpu_count() # 设置最大并发数

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务到线程池
            future_to_item = {executor.submit(func, item): item for item in items}

            # 使用 tqdm 显示进度条
            for future in tqdm(as_completed(future_to_item), total=len(items), desc='Processing'):
                try:
                    processed_item = future.result()
                    # 立即写入文件
                    if processed_item is not None:
                        outfile.write(processed_item)
                except Exception as e:
                    print(f"Error processing item: {e}")
                    raise e
    return output_path
