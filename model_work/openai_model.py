import openai
from openai import OpenAI
from typing import Dict, Any

class BaseOpenaiModel:
    """
    基础类，用于封装 OpenAI API 的调用逻辑。
    """

    def __init__(self, base_url: str,api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        初始化 OpenAI API 客户端。

        :param api_key: OpenAI 的 API 密钥。
        :param model_name: 使用的模型名称，默认为 "gpt-3.5-turbo"。
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = self.api_key

    def call(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        """
        调用 OpenAI API 完成文本生成任务。

        :param prompt: 输入的提示文本。
        :param max_tokens: 生成的最大 token 数量，默认为 4096。
        :param temperature: 控制生成文本的随机性，范围为 0.0 到 1.0，默认为 0.7。
        :param kwargs: 其他可选参数，例如 `stop`, `n` 等。
        :return: 包含 API 响应的字典。
        """
        try:
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return self._extract_response_text(response=response)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    def _extract_response_text(self, response: Dict[str, Any]) -> str:
        """
        从 OpenAI API 响应中提取生成的文本。

        :param response: OpenAI API 返回的响应字典。
        :return: 提取的生成文本。
        """
        if len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        else:
            raise ValueError("无法从响应中提取生成的文本，请检查 API 响应格式。")

# 示例使用
if __name__ == "__main__":
    import sys
    sys.path.append('./')
    from model_work.api_cfg import DEEPSEEK_BASE_URL,DEEPSEEK_API_KEY,DeepseekModels
    # 初始化模型
    BASE_URL =  DEEPSEEK_BASE_URL
    API_KEY = DEEPSEEK_API_KEY
    model_name = DeepseekModels.deepseekv3.MODEL_NAME
    model = BaseOpenaiModel(base_url=BASE_URL, api_key=API_KEY, model_name=model_name)
    # 调用 API
    prompt = """你是谁啊？"""
    response = model.call(prompt, max_tokens=2048, temperature=0.7)
    print(prompt ,"\n\n", response)