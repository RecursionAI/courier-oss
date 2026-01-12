import subprocess
import os
import uuid

import requests
from fastapi import HTTPException
from fastapi import Response
from starlette.responses import JSONResponse

from src.lora.helpers import create_lora_config, create_formatted_dataset, delete_dataset, evaluate, uc_header
from src.lora.models import LoraRequest


async def lora_train(request: LoraRequest, base_url):
    from datetime import datetime
    now = datetime.now()
    name = request.model_name.replace("/", "_")
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    new_adapter_path = f"src/lora/adapters/{request.company_name}/{request.product_name}/{name}/{now_str}"
    new_adapter_path = new_adapter_path.replace(" ", "_")
    new_adapter_path = new_adapter_path.replace(":", "_")
    try:
        create_formatted_dataset(model_name=request.model_name, dataset_name=request.dataset_name,
                                 api_key=request.api_key,
                                 val_size=request.val_size)
        file_path, config = create_lora_config(request.adapter_path, request.dataset_id, new_adapter_path)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Placeholder for CUDA-compatible training command (e.g., using unsloth or peft)
        command = [
            "python", "-m", "trl.commands.sft",
            "--config", "config.json",
        ]

        # Run LoRA training as a separate process
        # In a real CUDA environment, this would run the training
        # subprocess.run(command, check=True)

        # Run post-training evaluation (now async using vLLM)
        resp = await evaluate(request.model_name, new_adapter_path, config["max_seq_length"])
        
        return JSONResponse({"success": "success", "eval": resp}, status_code=200)
    except Exception as e:
        delete_dataset()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            r = requests.post(f"{base_url}/create-adapter/", headers={"Authorization": f"{request.api_key}"},
                              json={"model": f"{request.model_name}", "path": new_adapter_path, "display_name": f"{request.display_name}"})
            print(f"adapter path upload status: {r.status_code}")
        except Exception as e:
            print(f"Failed to upload adapter path: {e}")
        delete_dataset()
        print("////////////////////    Done    ///////////////////////")
