import gc
import uuid
from fastapi.responses import JSONResponse
from src.db.schemas import CourierModel
from typing import Optional, List, Any, Dict
import asyncio
import requests
import os
from PIL import Image
import base64
import io
import tempfile
import subprocess
import re

async def create_audio_response(request, model):
    """Simplified audio inference using model manager"""
    from src.model_manager.model_manager import model_manager

    temp_file = None
    try:
        # Extract audio from messages
        audio_data = None
        system_prompt = ""

        for msg in request.messages:
            role = msg.get("role")
            content = msg.get("content")

            # Extract system prompt if present
            if role == "system":
                if isinstance(content, str):
                    system_prompt = content
                elif isinstance(content, dict):
                    system_prompt = content.get("text", "")

            # Extract audio from message content
            if isinstance(content, dict):
                audio_obj = content.get("audio")
                if audio_obj:
                    if isinstance(audio_obj, bytes):
                        audio_data = audio_obj
                    if isinstance(audio_obj, str):
                        try:
                            response = requests.get(audio_obj, timeout=30)
                            response.raise_for_status()
                            audio_data = response.content
                        except Exception as e:
                            return JSONResponse({"error": f"Failed to download audio from URL: {str(e)}"},
                                                status_code=400)

        if not audio_data:
            return JSONResponse({"error": "No audio data found in messages."}, status_code=400)

        # Write audio data to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_data)
            temp_file = tmp.name

        # Build payload for vLLM
        payload = request.model_dump()
        payload["audio_file"] = temp_file
        if system_prompt:
             payload["system_prompt"] = system_prompt

        # Use model manager for inference (handles flex/static internally)
        result = await model_manager.inference(model, payload)

        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)

        if result.get("status_code") == 200:
            return result
        else:
            return JSONResponse(result, status_code=result.get("status_code", 500))

    except Exception as e:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)
        raise e


async def create_vision_response(request, model):
    """Simplified vision inference using model manager"""
    from src.model_manager.model_manager import model_manager

    temp_video_files = []
    try:
        processed_messages = []
        for msg in request.messages:
            role = msg.get("role")
            content = msg.get("content")

            text = ""
            images = []
            video_path = None

            if isinstance(content, str):
                text = content
            elif isinstance(content, dict):
                text = content.get("text", "")
                image_bytes_list = content.get("image_bytes")
                if image_bytes_list:
                    for ib in image_bytes_list:
                        try:
                            decoded_bytes = base64.b64decode(ib)
                            img = Image.open(io.BytesIO(decoded_bytes))
                            images.append(img)
                        except Exception as e:
                            print(f"Warning: Could not decode/open image: {e}")

                video_data = content.get("video")
                if video_data:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                            if isinstance(video_data, str) and (
                                    video_data.startswith("http://") or video_data.startswith("https://")):
                                response = requests.get(video_data, timeout=30)
                                response.raise_for_status()
                                tmp.write(response.content)
                            else:
                                tmp.write(video_data)
                            video_path = tmp.name
                            temp_video_files.append(video_path)
                    except Exception as e:
                        print(f"ERROR: Could not process video: {e}")

            processed_msg = {
                "role": role,
                "text": text,
                "images": images if images else None,
                "video": video_path
            }
            processed_messages.append(processed_msg)

        if not processed_messages:
            return JSONResponse({"error": "No messages provided for vision inference"}, status_code=400)

        payload = request.model_dump()
        payload["messages"] = processed_messages
        payload["video_num_frames"] = 30

        # Use model manager for inference (handles flex/static internally)
        result = await model_manager.inference(model, payload)

        if result.get("status_code") == 200:
            return result
        else:
            return JSONResponse(result, status_code=result.get("status_code", 500))

    except Exception as e:
        print(f"{e}")
        return JSONResponse({"error": f"Vision inference error: {str(e)}"}, status_code=500)
    finally:
        for vf in temp_video_files:
            try:
                if os.path.exists(vf):
                    os.unlink(vf)
            except Exception:
                pass


def needs_audio_conversion(data: bytes, content_type: str = "") -> bool:
    if data[:3] == b'ID3':
        return False
    if len(data) >= 2 and data[0] == 0xff and (data[1] & 0xe0) == 0xe0:
        return False
    return True


def convert_audio_to_mp3(audio_data: bytes) -> Optional[bytes]:
    input_temp = None
    output_temp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(audio_data)
            input_temp = f.name

        output_temp = tempfile.mktemp(suffix=".mp3")
        cmd = [
            "ffmpeg", "-y",
            "-i", input_temp,
            "-vn",
            "-acodec", "libmp3lame",
            "-q:a", "2",
            output_temp
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            return None
        with open(output_temp, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"Audio conversion error: {e}")
        return None
    finally:
        if input_temp and os.path.exists(input_temp):
            os.unlink(input_temp)
        if output_temp and os.path.exists(output_temp):
            os.unlink(output_temp)


def api_valid(api_key, courier_users) -> bool:
    valid_key = courier_users.read(key=f"{api_key}")
    if valid_key is not None:
        return valid_key.valid
    else:
        return False


def get_active_mem_gb() -> float:
    # generic version using psutil
    import psutil
    return (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024 ** 3)


def get_memory_stats() -> dict:
    """Get detailed memory statistics."""
    import psutil
    vm = psutil.virtual_memory()
    gb = 1024 ** 3
    
    stats = {
        "total_memory": f"{vm.total / gb:.2f} GB",
        "active_memory": f"{(vm.total - vm.available) / gb:.2f} GB",
        "memory_pressure": f"{vm.percent:.2f}%",
    }
    
    import torch
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        stats.update({
            "gpu_total_memory": f"{total_mem / gb:.2f} GB",
            "gpu_active_memory": f"{(total_mem - free_mem) / gb:.2f} GB",
        })
    
    return stats

# def can_load_model(workbench_model: CourierModel) -> bool:
#     gb_divisor = 1024 ** 3
