import re
import json

def markdown_extractor(output,tag='json'):
        """
        从模型输出中提取答案

        参数:
            output: 模型输出

        返回:
            提取的答案字符串
        """
        # 尝试从markdown代码块中提取最终答案
        pattern = rf'```{tag}\s*([\s\S]*?)\s*```'
        matches = re.findall(pattern, output)
        if matches:
            extracted_str = matches[-1].strip()
            return json.loads(extracted_str)
        return None
    
def text_extractor(output, tag='text'):
    pattern = rf'```{tag}\s*([\s\S]*?)\s*```'
    matches = re.findall(pattern, output)
    if matches:
        str = matches[-1].strip()
        return str
    return None
    
if __name__ == '__main__':
    output = '''
    ```json
    {
        "answer": "你好",
        "confidence": "你好"
    }
    ```
    '''
    print(markdown_extractor(output))