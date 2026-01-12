import asyncio
import os
import gc
import torch
from typing import Dict, List, Optional, Any, Union
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.lora.request import LoRARequest

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
        
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            trust_remote_code=trust_remote_code,
            enable_lora=bool(adapter_path),
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.lora_request = None
        if adapter_path:
            # We use a hash of the adapter path as a simple ID
            adapter_id = hash(adapter_path) & 0xFFFFFFFF
            self.lora_request = LoRARequest(f"adapter_{adapter_id}", adapter_id, adapter_path)

    async def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = payload.get("prompt")
        messages = payload.get("messages")
        sampling_params_dict = payload.get("sampling_params", {})
        
        # Extract common params
        temperature = payload.get("temperature", sampling_params_dict.get("temperature", 0.7))
        top_p = payload.get("top_p", sampling_params_dict.get("top_p", 0.9))
        max_tokens = payload.get("max_tokens", sampling_params_dict.get("max_tokens", 1024))
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **{k: v for k, v in sampling_params_dict.items() if k not in ["temperature", "top_p", "max_tokens"]}
        )
        
        request_id = os.urandom(8).hex()
        
        try:
            if messages:
                # Use vLLM's internal chat template support if available, 
                # or handle it via tokenizer if needed. 
                # AsyncLLMEngine.generate expects prompt or inputs.
                results_generator = self.engine.generate(
                    None, sampling_params, request_id, lora_request=self.lora_request, inputs={"messages": messages}
                )
            else:
                results_generator = self.engine.generate(
                    prompt, sampling_params, request_id, lora_request=self.lora_request
                )

            final_output = None
            try:
                # 3-minute timeout for hallucination detection / stall as per instructions
                async for request_output in asyncio.wait_for(results_generator, timeout=180.0):
                    final_output = request_output
            except asyncio.TimeoutError:
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
                "peak_memory": 0.0 # vLLM pre-allocates, so "peak" is usually constant
            }
        except Exception as e:
            return {"error": str(e), "status_code": 500}

    async def stop(self):
        # Attempt to free memory
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

    def ensure_pool(self, model: Any) -> vLLMModelPool:
        key = model.model_id
        if key in self._pools:
            return self._pools[key]
        
        pool = vLLMModelPool(
            model_name=model.file_path,
            adapter_path=model.adapter_path,
            tensor_parallel_size=model.tensor_parallel_size,
            pipeline_parallel_size=model.pipeline_parallel_size,
            gpu_memory_utilization=model.gpu_memory_utilization,
            max_model_len=model.max_model_len,
            max_num_seqs=model.max_num_seqs,
        )
        self._pools[key] = pool
        return pool

    async def remove_pool(self, model_id: str):
        pool = self._pools.pop(model_id, None)
        if pool:
            await pool.stop()

    def get_pool(self, model_id: str) -> Optional[vLLMModelPool]:
        return self._pools.get(model_id)

# Global registry instance
registry = vLLMModelPoolRegistry()
