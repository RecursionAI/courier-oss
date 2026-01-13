import time
import asyncio
import gc
import torch
import psutil
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, TypeVar
from src.db.schemas import CourierModel
from src.model_manager.vllm_model_pool import vLLMModelPool, registry

# Configure logging
logger = logging.getLogger("courier.model_manager")

T = TypeVar("T")

@dataclass
class ModelEntry:
    """Tracks model state and memory usage"""
    model: CourierModel
    loaded_at: float
    last_used: float
    # Memory tracking
    size_in_gb: float
    # The pool object
    pool: vLLMModelPool
    api_type: str = "flex"

    @property
    def is_expired(self) -> bool:
        """Check if model should be unloaded (5-minute TTL)"""
        if self.api_type.lower() == "flex":
            return time.time() - self.last_used > 300  # 5 minutes
        return False

    def mark_used(self):
        """Update last used timestamp"""
        self.last_used = time.time()


class ModelManager:
    def __init__(self):
        self.models: Dict[str, ModelEntry] = {}
        self.loaded_models: List[str] = []
        self.locks: Dict[str, asyncio.Lock] = {}
        self.last_errors: Dict[str, str] = {}

    async def get_model_entry(self, model_name: str) -> Optional[ModelEntry]:
        """Get model entry if exists and not expired"""
        entry = self.models.get(model_name)
        if entry:
            if not entry.is_expired:
                entry.mark_used()
                return entry
            else:
                # Use lock to prevent concurrent unloads of same model
                if model_name not in self.locks:
                    self.locks[model_name] = asyncio.Lock()
                async with self.locks[model_name]:
                    if model_name in self.models: # Double check
                        await self._unload_model(model_name)
                return None
        return None

    async def get_pool(self, model: CourierModel) -> Optional[vLLMModelPool]:
        """
        Get the appropriate pool for a model, loading it if necessary.
        """
        entry = await self.get_model_entry(model.name)

        if not entry:
            entry = await self.ensure_model_loaded(model)

        if entry:
            return entry.pool

        return None

    async def inference(self, model: CourierModel, payload: Any) -> Any:
        """
        Unified async inference method. Supports streaming.
        """
        pool = await self.get_pool(model)
        if not pool:
            error_msg = self.last_errors.get(model.name, "Failed to load model or get pool")
            return {"error": error_msg, "status_code": 500}

        try:
            if payload.get("stream"):
                return pool.infer_stream(payload)
            else:
                result = await pool.infer(payload)
                return result
        except Exception as e:
            logger.exception(f"Inference error for model {model.name}: {e}")
            return {"error": f"Inference error: {str(e)}", "status_code": 500}

    async def ensure_model_loaded(self, model: CourierModel) -> Optional[ModelEntry]:
        """Ensure model is loaded, handle memory management"""
        model_name = model.name
        
        if model_name not in self.locks:
            self.locks[model_name] = asyncio.Lock()
            
        async with self.locks[model_name]:
            existing = await self.get_model_entry(model_name)
            if existing:
                return existing

            required_gb = model.weights_gb
            available_gb = self.get_available_memory()
            logger.info(f"Loading model {model_name}. Available memory: {available_gb:.2f} GB, Required: {required_gb:.2f} GB")

            if available_gb >= (required_gb * 1.1):
                return await self._load_model(model)
            else:
                return await self._load_with_memory_management(model, available_gb, required_gb)

    async def _load_model(self, model: CourierModel) -> Optional[ModelEntry]:
        """Load model into memory"""
        try:
            # Phase 1.4: Non-blocking engine initialization
            pool = await asyncio.to_thread(registry.ensure_pool, model)

            entry = ModelEntry(
                model=model,
                loaded_at=time.time(),
                last_used=time.time(),
                pool=pool,
                size_in_gb=model.weights_gb,
                api_type=model.api_type,
            )

            self.models[model.name] = entry
            self.loaded_models.append(model.name)
            self.last_errors.pop(model.name, None)
            logger.info(f"Successfully loaded model: {model.name}")
            return entry
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to load model {model.name}: {error_msg}")
            self.last_errors[model.name] = error_msg
            return None

    async def _load_with_memory_management(self, model: CourierModel, available_gb: float, required_gb: float) -> Optional[ModelEntry]:
        """Handle memory management when OOM would occur"""
        models_to_unload = self._find_models_to_unload(required_gb, available_gb)

        if not models_to_unload:
            logger.warning(f"Could not find enough models to unload to free up {required_gb} GB")
        else:
            logger.info(f"Unloading models to free up memory: {models_to_unload}")

        for model_name in models_to_unload:
            await self._unload_model(model_name)

        new_available_gb = self.get_available_memory()
        if new_available_gb >= (required_gb * 1.05): # Use a small buffer
            return await self._load_model(model)
        else:
            error_msg = f"Insufficient memory after cleanup: {required_gb}GB needed, {new_available_gb:.2f}GB available"
            logger.error(error_msg)
            self.last_errors[model.name] = error_msg
            return None

    def _find_models_to_unload(self, required_gb: float, available_gb: float) -> List[str]:
        """Find models to unload using LRU policy. Protects static models."""
        # Only consider flex models for automatic unloading
        candidate_entries = [e for e in self.models.values() if e.api_type.lower() == "flex"]
        candidate_entries.sort(key=lambda x: x.last_used)

        total_free = available_gb
        models_to_unload = []

        # We want to have at least 10% buffer
        target_free = required_gb * 1.1

        if total_free < target_free:
            for entry in candidate_entries:
                if total_free >= target_free:
                    break
                models_to_unload.append(entry.model.name)
                total_free += entry.size_in_gb

        return models_to_unload

    async def _unload_model(self, model_name: str):
        """Unload a model from memory"""
        if model_name in self.models:
            entry = self.models[model_name]
            # registry now expects the model object for deduplication/ref-counting
            await registry.remove_pool(entry.model)
            del self.models[model_name]
            self.loaded_models = [name for name in self.loaded_models if name != model_name]
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {model_name}")

    def get_available_memory(self) -> float:
        """Get available memory in GB"""
        # Phase 2.2: CUDA Initialization Safety - Use nvidia-smi if available to avoid early CUDA init
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            # nvidia-smi might return multiple lines if multiple GPUs
            free_mem_mb = float(result.stdout.strip().split('\n')[0])
            return free_mem_mb / 1024.0
        except Exception:
            # Fallback to torch if nvidia-smi fails or is not available
            if torch.cuda.is_available():
                try:
                    free_mem, _ = torch.cuda.mem_get_info()
                    return free_mem / (1024 ** 3)
                except Exception:
                    pass
            
            return psutil.virtual_memory().available / (1024 ** 3)

    async def cleanup_expired_models(self):
        """Clean up all expired models"""
        expired = [name for name, entry in self.models.items() if entry.is_expired]
        for model_name in expired:
            logger.info(f"Model {model_name} expired. Unloading...")
            await self._unload_model(model_name)

    async def monitor_health(self):
        """Check health of all loaded models and unload if unhealthy"""
        for name, entry in list(self.models.items()):
            is_healthy = await entry.pool.check_health()
            if not is_healthy:
                logger.error(f"Model {name} is unhealthy. Unloading for recovery...")
                await self._unload_model(name)

    async def shutdown(self):
        """Gracefully shutdown all loaded models"""
        logger.info("Shutting down ModelManager and all vLLM engines...")
        model_names = list(self.models.keys())
        for name in model_names:
            await self._unload_model(name)
        logger.info("ModelManager shutdown complete.")


# Global model manager instance
model_manager = ModelManager()
