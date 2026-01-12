import json
import os
import shutil
import requests
import gc
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
from .evaluator import AIOutputEvaluator

uc_header = {"Authorization": "uce_asdfawea_94923949"}
base_url = "https://launchco.uc.r.appspot.com/api/uce"

def create_formatted_dataset(model_name, api_key, dataset_name, val_size: float):
    """Format training pairs into JSONL format with chat template and save to file."""
    conversation_data = requests.get(f"{base_url}/get-courier-conversations-for-user/",
                              headers={"Authorization": f"{api_key}"}, data={"dataset_name": dataset_name})
    data_dir = "dataset"
    os.makedirs(data_dir, exist_ok=True)

    conversations = conversation_data.json()["response"]
    dataset = Dataset.from_list(conversations)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def apply_chat_formatting(example):
        if "text" in example:
            return {"text": example["text"]}
        messages = []
        if "system" in example and example["system"]:
            messages.append({"role": "system", "content": example["system"]})
        if "prompt" in example and "completion" in example:
            messages.append({"role": "user", "content": example["prompt"]})
            messages.append({"role": "assistant", "content": example["completion"]})
        elif "messages" in example:
            messages = example["messages"]
        else:
            raise ValueError("Training pairs must have 'input'/'output' fields or 'messages' field")

        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": formatted_text}

    formatted_dataset = dataset.map(apply_chat_formatting)
    if 'text_item_id' in formatted_dataset.column_names:
        formatted_dataset = formatted_dataset.select_columns(['text'])
    shuffled_dataset = formatted_dataset.shuffle(seed=42)
    total_size = len(shuffled_dataset)
    val_size_int = int(val_size * total_size)
    validation_dataset = shuffled_dataset.select(range(val_size_int))
    train_dataset = shuffled_dataset.select(range(val_size_int, total_size))

    train_filename = os.path.join(data_dir, "train.jsonl")
    val_filename = os.path.join(data_dir, "valid.jsonl")
    train_dataset.to_json(train_filename)
    validation_dataset.to_json(val_filename)
    return train_filename, val_filename

def delete_dataset():
    """Delete the data directory and all its contents."""
    data_dir = "dataset"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    if os.path.exists("config.yaml"):
        os.remove("config.yaml")

def create_lora_config(resume_adapter: str, dataset_id, new_adapter_path):
    try:
        response = requests.get(f"{base_url}/get-lora-config/{dataset_id}/", headers=uc_header)
        config = response.json()
        # CUDA trainer config (e.g. for peft/trl)
        config_data = {
            "model_name_or_path": config['model'],
            "dataset_name": "dataset",
            "output_dir": new_adapter_path,
            "max_steps": config['iters'],
            "per_device_train_batch_size": config['batch_size'],
            "learning_rate": config['learning_rate'],
            "max_seq_length": config['max_seq_length'],
            "lora_rank": config['rank'],
            "lora_alpha": config['alpha'],
            "lora_dropout": config['dropout'],
        }
        file_path = "config.json"
        with open(file_path, "w") as f:
            json.dump(config_data, f, indent=4)
        return file_path, config
    except Exception as e:
        return {"error": str(e)}

async def evaluate(model_name: str, adapter_path: str, context_window: int) -> str:
    """
    Evaluation using vLLM via ModelManager.
    """
    from src.model_manager.vllm_model_pool import vLLMModelPool
    
    # Create a temporary pool for evaluation
    pool = vLLMModelPool(
        model_name=model_name,
        adapter_path=adapter_path,
        max_model_len=context_window
    )
    
    try:
        stop_sequences = ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>", "[INST]", "[/INST]"]
        prompts: List[str] = []
        references: List[str] = []

        with open("dataset/valid.jsonl", "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= 10:
                    break
                item = json.loads(line)
                messages = item.get("messages", [])
                # In a real scenario we'd use a tokenizer here, 
                # but vLLM handles messages if we pass them.
                ref = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
                prompts.append(messages)
                references.append(ref)

        eval_pairs: List[Tuple[str, str]] = []
        for i, msgs in enumerate(prompts):
            payload = {
                "messages": msgs,
                "temperature": 0.0,
                "max_tokens": context_window
            }
            result = await pool.infer(payload)
            gen = result.get("content", "").strip()
            eval_pairs.append((gen, references[i]))

        evaluator = AIOutputEvaluator()
        result = evaluator.batch_evaluate(eval_pairs)
        json_data = evaluator.export_results(result, format="json")
        with open("eval.json", "w", encoding="utf-8") as f:
            f.write(json_data)
        return json_data
    finally:
        await pool.stop()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
