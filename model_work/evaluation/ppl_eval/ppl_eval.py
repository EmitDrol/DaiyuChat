import json
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{cur_dir}/../../../')

from myutils.fomat_time import formatted_time
from sys_prompts.DaiyuLin_sp import prompt as daiyu_sys_prompt
import torch
import pandas as pd
from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

def calculate_ppl(context, response, model, tokenizer, device='cuda', max_length=2048):
    try:
        # Tokenize context and response separately
        context_enc = tokenizer.encode(context, add_special_tokens=False)
        response_enc = tokenizer.encode(response, add_special_tokens=False)
        
        # Truncate context to leave space for response
        available_length = max_length - len(response_enc)
        context_enc = context_enc[-available_length:] if available_length > 0 else []
        
        # Combine input sequences
        input_ids = context_enc + response_enc
        
        # Create labels (-100 for context, actual tokens for response)
        labels = [-100] * len(context_enc) + response_enc
        
        # Convert to tensors
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        labels = torch.tensor([labels], dtype=torch.long).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
        
        return torch.exp(outputs.loss).item()
    
    except Exception as e:
        print(f"Error processing text: {context[-50:]}... | {str(e)}")
        return None

def ppl_eval(dialogs, model_path="gpt2", 
            output_file=f"model_work/evaluation/eval_report/ppl_results_{formatted_time()}.xlsx",sys_promt=None,test_data_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,trust_remote_code=True)
    
    # 将模型移动到 GPU
    model.to(device)
    
    model.eval()
    
    results = []
    total_assistants = sum(1 for dialog in dialogs for msg in dialog if msg['role'] == 'assistant')
    valid_count = 0
    with tqdm(total=total_assistants, desc="Evaluating Dialogs", unit="reply", ncols=120) as progress_bar:
        for dialog_id, dialog in enumerate(dialogs):
            if sys_promt:
                dialog = [{"role": "system", "content": sys_promt}] + dialog
            for msg_idx, msg in enumerate(dialog):
                if msg['role'] == 'assistant':
                    try:
                        # Build context and response
                        context = tokenizer.apply_chat_template(dialog[:msg_idx], tokenize=False, add_generation_prompt=True)
                        response = msg['content'] 
                        
                        # Calculate PPL for response given context
                        ppl = calculate_ppl(
                            context,
                            response,
                            model,
                            tokenizer,
                            device,
                            max_length=model.config.max_position_embeddings
                        )
                        
                        if not context or not response:
                            print(f"Error processing dialog {dialog_id + 1}: Context or response is empty")
                            print(dialog)
                            raise ValueError("Context or response is empty")
                        
                        if ppl is not None:
                            results.append({
                                "ID": dialog_id + 1,
                                "index": len(results) + 1,
                                "context": context,
                                "gold_response": response,
                                "ppl": round(ppl, 2)
                            })
                            valid_count += 1
                    except Exception as e:
                        print(f"Error processing dialog {dialog_id + 1}: {str(e)}")
                    progress_bar.update(1)
    
    
    progress_bar.close()
    
    # 释放 GPU 资源
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存主结果
    details_file = output_file.replace(".xlsx", ".jsonl")
    with open(details_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 保存元数据
    meta_output_file = output_file.replace(".xlsx", "_meta.xlsx")
    meta_data = {
        "model_path": model_path,
        "test_data_path": test_data_path,
        "num_dialogs": len(dialogs),
        "total_assistants": total_assistants,
        "valid_count": valid_count,
        "invalid_count": total_assistants - valid_count,
        "avg_ppl": round(sum(result["ppl"] for result in results) / valid_count, 2) if valid_count > 0 else 0,
    }
    
    meta_df = pd.DataFrame([meta_data])
    meta_df.to_excel(meta_output_file, index=False)
    print(f"✅ 评测结果已保存到 {meta_output_file}")


def run_ppl_eval(test_data_path, model_paths:list[str],  sys_promt=daiyu_sys_prompt):
    time_stamp = formatted_time()
    
    # 验证模型路径
    for model_path in model_paths:
        if not os.path.exists(model_path):
            raise ValueError(f"模型路径 {model_path} 不存在")
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        file_data = json.load(f)
    
    test_data = [dialog["scene_messages"] for dialog in file_data]
    
    for model_path in model_paths:
        ppl_eval(test_data, model_path, output_file=f"/cpfs01/shared/llm_ddd/chenyongkang/DaiyuChat/model_work/evaluation/eval_report/ppl_{time_stamp}/ppl_results_{os.path.basename(model_path).replace('.json','')}_{os.path.basename(test_data_path)}.xlsx", sys_promt=sys_promt,test_data_path=test_data_path)

# 使用示例
if __name__ == "__main__":
    repo_path = '/cpfs01/shared/llm_ddd/chenyongkang/DaiyuChat/'
    run_ppl_eval(test_data_path=f'{repo_path}data_work/data/train_data/data_v4_test.json',model_paths=[f'{repo_path}model_work/ckpts/qwen3_32b'])