import asyncio
import os
import gc
import torch
import logging
import hashlib
import json
import re
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.lora.request import LoRARequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("courier.vllm_pool")

class vLLMModelPool:
    def __init__(
            self,
            model_name: str,
            adapter_path: Optional[str] = None,
            tensor_parallel_size: int = 1,
            pipeline_parallel_size: int = 1,
            gpu_memory_utilization: float = 0.9,
            max_model_len: Optional[int] = None,
            max_num_seqs: int = 256,
            trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        
        # Load tokenizer for chat template support
        from transformers import AutoTokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {e}")
            raise e

        # Phase 2.2: Hardware-aware configuration
        dtype = "auto"
        try:
            # Use nvidia-smi to check compute capability without initializing CUDA
            import subprocess
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True
            )
            if res.returncode == 0:
                major = int(res.stdout.strip().split('.')[0])
                if major < 8:
                    dtype = "float16"
                    logger.info(f"Hardware support for bfloat16 not detected (Compute Capability {major}). Falling back to float16.")
            elif torch.cuda.is_available():
                # Fallback if nvidia-smi is not available
                major, _ = torch.cuda.get_device_capability()
                if major < 8:
                    dtype = "float16"
                    logger.info(f"Hardware support for bfloat16 not detected (Compute Capability {major}). Falling back to float16.")
        except Exception as e:
            logger.warning(f"Warning during hardware detection: {e}")

        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            enable_lora=bool(adapter_path),
        )
        
        try:
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        except ValueError as e:
            error_msg = str(e)
            # Handle the specific case where max_model_len is too large for available VRAM
            if "estimated maximum model length is" in error_msg:
                match = re.search(r"estimated maximum model length is (\d+)", error_msg)
                if match:
                    suggested_len = int(match.group(1))
                    logger.warning(
                        f"Model {model_name} failed to load with max_model_len={max_model_len or 'auto'}. "
                        f"Retrying with suggested max_model_len={suggested_len} based on available VRAM."
                    )
                    engine_args.max_model_len = suggested_len
                    self.engine = AsyncLLMEngine.from_engine_args(engine_args)
                else:
                    raise e
            else:
                raise e
        self.lora_request = None
        if adapter_path:
            # We use a hash of the adapter path as a simple ID
            adapter_id = int(hashlib.md5(adapter_path.encode()).hexdigest(), 16) & 0xFFFFFFFF
            self.lora_request = LoRARequest(f"adapter_{adapter_id}", adapter_id, adapter_path)

    async def _prepare_params(self, payload: Dict[str, Any]):
        prompt = payload.get("prompt")
        messages = payload.get("messages")
        
        # Defensive parameter parsing
        sampling_params_dict = payload.get("sampling_params") or {}
        if not isinstance(sampling_params_dict, dict):
            sampling_params_dict = {}

        # Extract common params with fallback to payload-level then defaults
        temperature = payload.get("temperature", sampling_params_dict.get("temperature", 0.7))
        top_p = payload.get("top_p", sampling_params_dict.get("top_p", 0.9))
        max_tokens = payload.get("max_tokens", sampling_params_dict.get("max_tokens", 1024))

        sampling_params = SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
            **{k: v for k, v in sampling_params_dict.items() if k not in ["temperature", "top_p", "max_tokens"]}
        )

        multi_modal_data = None
        if messages:
            # Use chat template to format messages into a prompt string
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Phase 3: Multi-modal support (Vision)
            images = []
            for msg in messages:
                if isinstance(msg, dict) and msg.get("images"):
                    images.extend(msg["images"])
            
            if images:
                # vLLM expects multi_modal_data for vision models
                multi_modal_data = {"image": images}
        
        return prompt, sampling_params, multi_modal_data

    async def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_id = os.urandom(8).hex()
        try:
            prompt, sampling_params, multi_modal_data = await self._prepare_params(payload)
            
            if not prompt:
                return {"error": "No prompt or messages provided", "status_code": 400}

            results_generator = self.engine.generate(
                prompt, sampling_params, request_id, 
                lora_request=self.lora_request,
                multi_modal_data=multi_modal_data
            )

            final_output = None
            try:
                # 3-minute timeout for hallucination detection / stall as per instructions
                async with asyncio.timeout(180.0):
                    async for request_output in results_generator:
                        final_output = request_output
            except (asyncio.TimeoutError, TimeoutError):
                logger.error(f"Inference timed out for request {request_id}")
                return {"error": "Inference timed out (hallucination detection)", "status_code": 504}

            if final_output is None:
                return {"error": "No output generated", "status_code": 500}

            text = final_output.outputs[0].text
            prompt_tokens = len(final_output.prompt_token_ids)
            completion_tokens = len(final_output.outputs[0].token_ids)

            return {
                "content": text,
                "status_code": 200,
                "prompt_tokens": prompt_tokens,
                "generation_tokens": completion_tokens,
                "peak_memory": 0.0
            }
        except Exception as e:
            logger.exception(f"Inference error for request {request_id}: {e}")
            return {"error": str(e), "status_code": 500}

    async def infer_stream(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        request_id = os.urandom(8).hex()
        try:
            prompt, sampling_params, multi_modal_data = await self._prepare_params(payload)
            if not prompt:
                yield f"data: {json.dumps({'error': 'No prompt or messages provided', 'status_code': 400})}\n\n"
                return

            results_generator = self.engine.generate(
                prompt, sampling_params, request_id, 
                lora_request=self.lora_request,
                multi_modal_data=multi_modal_data
            )

            last_pos = 0
            prompt_tokens = 0
            generation_tokens = 0
            
            async with asyncio.timeout(180.0):
                async for request_output in results_generator:
                    text = request_output.outputs[0].text
                    delta = text[last_pos:]
                    last_pos = len(text)
                    
                    if request_output.finished:
                        prompt_tokens = len(request_output.prompt_token_ids)
                        generation_tokens = len(request_output.outputs[0].token_ids)
                    
                    # Standard SSE format: data: <json>\n\n
                    yield f"data: {json.dumps({
                        'text': delta,
                        'finished': request_output.finished,
                        'prompt_tokens': prompt_tokens if request_output.finished else None,
                        'generation_tokens': generation_tokens if request_output.finished else None,
                    })}\n\n"
        except (asyncio.TimeoutError, TimeoutError):
            logger.error(f"Stream timed out for request {request_id}")
            yield f"data: {json.dumps({'error': 'Inference timed out', 'status_code': 504})}\n\n"
        except Exception as e:
            logger.exception(f"Stream error for request {request_id}: {e}")
            yield f"data: {json.dumps({'error': str(e), 'status_code': 500})}\n\n"

    async def check_health(self) -> bool:
        """Ping the engine to check if it's still alive"""
        try:
            # We use a very simple generation request as a health check
            sampling_params = SamplingParams(max_tokens=1)
            request_id = "health-check-" + os.urandom(4).hex()
            results_generator = self.engine.generate("ping", sampling_params, request_id)
            async with asyncio.timeout(5.0):
                async for _ in results_generator:
                    break
            return True
        except Exception as e:
            logger.error(f"Health check failed for model {self.model_name}: {e}")
            return False

    async def stop(self):
        # Attempt to free memory
        logger.info(f"Stopping vLLM engine for {self.model_name}")
        if hasattr(self, 'engine'):
            # vLLM doesn't have a formal shutdown yet for AsyncLLMEngine in all versions,
            # but we can try to trigger cleanup.
            del self.engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



class vLLMModelPoolRegistry:
    def __init__(self):
        self._pools: Dict[str, vLLMModelPool] = {}
        self._refs: Dict[str, int] = {}

    def _get_key(self, model: Any) -> str:
        # Deduplicate based on model path and engine configuration
        import hashlib
        import json
        config = {
            "file_path": model.file_path,
            "adapter_path": model.adapter_path,
            "gpu_memory_utilization": model.gpu_memory_utilization,
            "max_model_len": model.max_model_len,
            "max_num_seqs": model.max_num_seqs,
        }
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def ensure_pool(self, model: Any) -> vLLMModelPool:
        key = self._get_key(model)
        if key in self._pools:
            self._refs[key] += 1
            return self._pools[key]

        import dotenv
        dotenv.load_dotenv()
        tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", 1))
        pipeline_parallel_size = int(os.getenv("PIPELINE_PARALLEL_SIZE", 1))

        pool = vLLMModelPool(
            model_name=model.file_path,
            adapter_path=model.adapter_path,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            gpu_memory_utilization=model.gpu_memory_utilization,
            max_model_len=model.max_model_len,
            max_num_seqs=model.max_num_seqs,
        )
        self._pools[key] = pool
        self._refs[key] = 1
        return pool

    async def remove_pool(self, model: Any):
        # We now expect the model object or a similar object with file_path etc.
        # If it's a string, we assume it's an old-style model_id and try to find it (for compatibility if needed)
        if isinstance(model, str):
             # Fallback: this shouldn't happen after full migration but good for safety
             key = model 
        else:
            key = self._get_key(model)
            
        if key in self._pools:
            self._refs[key] -= 1
            if self._refs[key] <= 0:
                pool = self._pools.pop(key)
                self._refs.pop(key, None)
                await pool.stop()

    def get_pool(self, model: Any) -> Optional[vLLMModelPool]:
        if isinstance(model, str):
            return self._pools.get(model)
        return self._pools.get(self._get_key(model))


# Global registry instance
registry = vLLMModelPoolRegistry()
