import math
import openai
from openai import OpenAI
from typing import Dict, Any

class BaseOpenaiModel:
    """
    基础类，用于封装 OpenAI API 的调用逻辑。
    """
    TEMPLATE_MAP = {
    "r1": {"chat_template":"<｜begin▁of▁sentence｜><｜User｜>{input}<｜Assistant｜><think>\n","stop_words":["<｜end▁of▁sentence｜>"]}, # r1 new chat template
    "qwen": {"chat_template":"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n","stop_words":["<|im_end|>", "<|endoftext|>"]}, # default qwen template
    "internthinker":{"chat_template":"<|im_start|>system\nYou are an expert reasoner with extensive experience in mathematical and code competitions. You approach problems through systematic thinking and rigorous reasoning. Your response should reflect deep understanding and precise logical thinking, making your solution path and reasoning clear to others. Please put your thinking process within <think>...</think> tags.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n","stop_words":["<|im_end|>", "<|endoftext|>"]},
    "chatml":{"chat_template":"<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n","stop_words":["<|im_end|>", "<|endoftext|>"]}, # No sys prompt chatml
}


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

    def call(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7, api_mode='chat_completion',template_mode='chatml', sys_prompt=None, **kwargs) -> str:
        """
        调用 OpenAI API 完成文本生成任务。

        :param prompt: 输入的提示文本。
        :param max_tokens: 生成的最大 token 数量，默认为 4096。
        :param temperature: 控制生成文本的随机性，范围为 0.0 到 1.0，默认为 0.7。
        :param kwargs: 其他可选参数，例如 `stop`, `n` 等。
        :return: 包含 API 响应的字典。
        """
        template = self.TEMPLATE_MAP[template_mode]
        chat_template = template["chat_template"]
        stop_words = template["stop_words"]
        
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        
        if api_mode == "chat_completion":
            messages = [{"role": "user", "content": prompt}]
            if sys_prompt:
                messages.insert(0, {"role": "system", "content": sys_prompt})
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            output = response.choices[0].message.content
        elif api_mode == "completion":
            response = client.completions.create(
                model=self.model_name,
                prompt=chat_template.format(input=prompt),  # Use templated prompt
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_words if stop_words else None,
                **kwargs
            )
            output = response.choices[0].text
        else:
            raise ValueError("Invalid API mode")
        return output


    def messages_call(self, messages: list, max_tokens: int = 4096, temperature: float = 0.7,sys_prompt=None ,**kwargs) -> str:
        """
        调用 OpenAI API 完成文本生成任务。
        :param messages: 输入的提示文本。
        :param max_tokens: 生成的最大 token 数量，默认为 4096。
        :param temperature: 控制生成文本的随机性，范围为 0.0 到 1.0，默认为 0.7。
        :param kwargs: 其他可选参数，例如 `stop`, `n` 等。
        :return: 包含 API 响应的字典。
        """
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages if not sys_prompt else [{"role": "system", "content": sys_prompt}] + messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response.choices[0].message.content
    
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

    def calculate_generation_perplexity(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, template='qwen', **kwargs) -> (str, float):
            """
            计算生成文本的困惑度 (PPL) 并返回生成文本和困惑度值。

            :param prompt: 输入的提示文本。
            :param max_tokens: 生成的最大 token 数量，默认为 2048。
            :param temperature: 控制生成文本的随机性，范围为 0.0 到 1.0，默认为 0.7。
            :param kwargs: 其他传递给 API 的参数。
            :return: 生成的文本和对应的困惑度值。
            """
            # 调用 OpenAI API 获取生成的文本和 log probabilities
            chat_template = self.TEMPLATE_MAP[template]["chat_template"]
            stop_words=self.TEMPLATE_MAP[template]["stop_words"]
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            response = client.completions.create(
                model=self.model_name,
                prompt=chat_template.format(input=prompt),  # Use templated prompt
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_words,
                logprobs=1,  # 获取生成 token 的 log probabilities
                echo=False,  # 不返回输入的 token 和其概率
                **kwargs
            )

            # 解析返回的 log probabilities
            import pprint
            pprint.pprint(response)
            log_probs = response.choices[0].logprobs.token_logprobs
            tokens = response.choices[0].logprobs.tokens

            # 累加 log probabilities
            total_log_prob = sum(log_prob for log_prob in log_probs if log_prob is not None)
            total_token_count = len(log_probs)

            # 计算困惑度
            avg_log_prob = total_log_prob / total_token_count if total_token_count > 0 else 0
            perplexity = math.exp(-avg_log_prob) if avg_log_prob < 0 else float('inf')

            # 提取生成的文本
            generated_text = response.choices[0].text.strip()

            return generated_text, perplexity

# 示例使用
if __name__ == "__main__":
    import sys
    sys.path.append('./')
    from model_work.api_cfg import DEEPSEEK_BASE_URL,DEEPSEEK_API_KEY,DeepseekModels
    # 初始化模型
    # BASE_URL =  DEEPSEEK_BASE_URL
    # API_KEY = DEEPSEEK_API_KEY
    # model_name = DeepseekModels.deepseekv3.MODEL_NAME
    API_KEY = "empty"
    BASE_URL = "http://10.130.5.11:8000/v1"
    model_name = 'daiyu_20250426_042114'
    model = BaseOpenaiModel(base_url=BASE_URL, api_key=API_KEY, model_name=model_name)
    # 调用 API
    prompt = """你是谁啊？"""
    response = model.calculate_generation_perplexity(prompt, max_tokens=2048, temperature=0.7,template='qwen')
    print(prompt ,"\n\n", response)