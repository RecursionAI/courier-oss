# This module wraps the vLLM AsyncLLMEngine to provide high-performance inference.

import asyncio
import os
import gc
import torch
import logging
import hashlib
import json
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.lora.request import LoRARequest

from src.db.schemas import CourierModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("courier.vllm_pool")


def calculate_gpu_utilization(
        max_model_len: int,
        available_vram_gb: float,
        model_weight_gb: float,
        buffer_gb: float = 1.0
) -> float:
    """
    Calculates the optimal gpu_memory_utilization based on model size and context window.

    Args:
        max_model_len: Requested context window.
        available_vram_gb: Total VRAM available on the device.
        model_weight_gb: Estimated size of model weights in GB.
        buffer_gb: Safety buffer in GB for overhead/activations.
    """
    # 1. Estimate KV Cache size (Rule of thumb: 1GB per 10k tokens for ~7B models)
    # Heuristic: ~200MB per 1024 tokens.
    kv_cache_gb = (max_model_len / 1024) * 0.2

    # 2. Total required memory
    total_needed_gb = model_weight_gb + kv_cache_gb + buffer_gb

    # 3. Calculate utilization ratio relative to total VRAM
    utilization = total_needed_gb / available_vram_gb

    # Constraints:
    # Minimum 0.1 to avoid errors,
    # Maximum 0.95 to leave room for the OS/system.
    util = max(0.1, min(0.95, utilization))
    print(f"max_memory_utilization: {util}")
    print("=" * 20)
    return util


class vLLMModelPool:
    def __init__(
            self,
            model_name: str,
            adapter_path: Optional[str] = None,
            tensor_parallel_size: int = 1,
            pipeline_parallel_size: int = 1,
            max_model_len: int = 6000,
            max_num_seqs: int = 1,
            trust_remote_code: bool = True,
            model_weight_gb: float = 4.0,
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

        # 1. HARDWARE AUTO-DETECTION:
        # During initialization, the pool checks the GPU's Compute Capability.
        # It automatically falls back to `float16` if `bfloat16` isn't supported by the hardware.
        # Phase 2.2: Hardware-aware configuration
        dtype = "auto"
        try:
            if torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability()
                if major < 8:
                    dtype = "float16"
                    logger.info(
                        f"Hardware support for bfloat16 not detected (Compute Capability {major}). Falling back to float16.")
        except Exception as e:
            logger.warning(f"Warning during hardware detection: {e}")

        # Calculate dynamic GPU utilization
        if torch.cuda.is_available():
            try:
                _, total_mem = torch.cuda.mem_get_info()
                total_mem_gb = total_mem / (1024 ** 3)
            except Exception as e:
                logger.warning(f"Failed to get GPU memory info: {e}. Falling back to 16GB.")
                total_mem_gb = 16.0
        else:
            total_mem_gb = 16.0

        dynamic_utilization = calculate_gpu_utilization(
            max_model_len=max_model_len,
            available_vram_gb=total_mem_gb,
            model_weight_gb=model_weight_gb
        )
        logger.info(f"Dynamic GPU utilization for {model_name} calculated: {dynamic_utilization:.4f} "
                    f"(Weights: {model_weight_gb}GB, Context: {max_model_len})")

        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            gpu_memory_utilization=dynamic_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            enable_lora=bool(adapter_path),
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        # 2. LORA INTEGRATION:
        # If an `adapter_path` is provided, the pool configures vLLM to use LoRA.
        # Adapters are identified by a hash of their path for efficient deduplication.
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

        if messages:
            # Use chat template to format messages into a prompt string
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        return prompt, sampling_params

    # 3. STREAMING & TIMEOUTS:
    # `infer` and `infer_stream` include a 3-minute watchdog timer 
    # to detect and recover from potential model stalls or "infinite" hallucinations.
    async def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_id = os.urandom(8).hex()
        try:
            prompt, sampling_params = await self._prepare_params(payload)

            if not prompt:
                return {"error": "No prompt or messages provided", "status_code": 400}

            results_generator = self.engine.generate(
                prompt, sampling_params, request_id, lora_request=self.lora_request
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
            prompt, sampling_params = await self._prepare_params(payload)
            if not prompt:
                yield json.dumps({"error": "No prompt or messages provided", "status_code": 400})
                return

            results_generator = self.engine.generate(
                prompt, sampling_params, request_id, lora_request=self.lora_request
            )

            last_pos = 0
            async with asyncio.timeout(180.0):
                async for request_output in results_generator:
                    text = request_output.outputs[0].text
                    delta = text[last_pos:]
                    last_pos = len(text)

                    yield json.dumps({
                        "text": delta,
                        "finished": request_output.finished,
                        "prompt_tokens": len(request_output.prompt_token_ids) if request_output.finished else None,
                        "generation_tokens": len(
                            request_output.outputs[0].token_ids) if request_output.finished else None,
                    })
        except (asyncio.TimeoutError, TimeoutError):
            logger.error(f"Stream timed out for request {request_id}")
            yield json.dumps({"error": "Inference timed out", "status_code": 504})
        except Exception as e:
            logger.exception(f"Stream error for request {request_id}: {e}")
            yield json.dumps({"error": str(e), "status_code": 500})

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


# 4. REGISTRY PATTERN:
# The `vLLMModelPoolRegistry` ensures that if multiple users request 
# the same model config, they share the same underlying engine instance (Ref Counting).
class vLLMModelPoolRegistry:
    def __init__(self):
        self._pools: Dict[str, vLLMModelPool] = {}
        self._refs: Dict[str, int] = {}

    def _get_key(self, model: CourierModel) -> str:
        # Deduplicate based on model path and engine configuration
        import hashlib
        import json
        config = {
            "file_path": model.file_path,
            "adapter_path": model.adapter_path,
            "weights_gb": model.weights_gb,
            "max_model_len": model.context_window,
            "max_num_seqs": model.instances,
        }
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def ensure_pool(self, model: CourierModel) -> vLLMModelPool:
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
            max_model_len=model.context_window,
            max_num_seqs=model.instances,
            model_weight_gb=model.weights_gb,
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
