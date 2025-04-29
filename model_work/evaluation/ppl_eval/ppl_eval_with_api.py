import sys
sys.path.append('.')

from openai import OpenAI
from model_work.api_cfg import DEEPSEEK_API_KEY,DEEPSEEK_BASE_URL,DeepseekModels

client = OpenAI(api_key=DEEPSEEK_API_KEY,base_url='https://api.deepseek.com/beta')

def get_logprobs(prompt,api_mode='chat_completion'):
    # 调用 OpenAI API 获取 logprobs
    
    resp = None
    logprobs = None
    if api_mode=='chat_completion':
        response = client.chat.completions.create(
            model=DeepseekModels.deepseekv3.MODEL_NAME,  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=16,                
            logprobs=True,       
        )
        resp = response.choices[0].message.content
        logprobs_data = response.choices[0].logprobs
        logprobs = [logprob.logprob for logprob in logprobs_data.content]
    elif api_mode=='completions':
        response = client.completions.create(
            model=DeepseekModels.deepseekv3.MODEL_NAME,  
            prompt=prompt,
            temperature=0.5,
            max_tokens=1,             
            logprobs=1,       
        )
        resp  = response.choices[0].text
        logprobs = response.choices[0].logprobs.token_logprobs 

    
    return resp,logprobs

# 示例 prompt
prompt = "什么是美人鱼？"

# 获取 logprobs
resp, logprobs = get_logprobs(prompt,api_mode='completions')
print('=' * 100)
print("Response:", resp)
print('=' * 100)
print("Logprobs:", logprobs)