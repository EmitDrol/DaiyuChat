import torch
import pandas as pd
from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModelForCausalLM

def calculate_ppl(text, model, tokenizer, device='cuda'):
    try:
        inputs = tokenizer(text, return_tensors='pt', 
                         truncation=True, 
                         padding=False).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
        return torch.exp(outputs.loss).item()
    except Exception as e:
        print(f"Error processing text: {text[:20]}... | {str(e)}")
        return None  # 标记无效值

def ppl_eval(text_list, model_name="gpt2", 
                 output_file="ppl_results.xlsx"):
    # 初始化设备和模型 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # 准备进度条
    results = []
    progress_bar = tqdm(total=len(text_list), 
                      desc="Processing", 
                      unit="text",
                      ncols=100)
    
    # 逐条处理文本 
    for idx, text in enumerate(text_list):
        ppl = calculate_ppl(text, model, tokenizer, device)
        if ppl is not None:
            results.append({
                "Index": idx+1,
                "Text": text,
                "Perplexity": round(ppl, 2)
            })
        progress_bar.update(1)  # 更新进度条 
    
    progress_bar.close()
    
    # 保存到Excel 
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"✅ 结果已保存到 {output_file}")

if __name__ == "__main__":
    texts = [
        "This is the first test sentence.",
        "Another example text for perplexity calculation.",
        "Natural language processing is fascinating!"
    ]
    
    ppl_eval(texts, output_file="ppl_results.xlsx")