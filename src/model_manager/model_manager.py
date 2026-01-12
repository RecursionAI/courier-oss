import time
import asyncio
import gc
import torch
import psutil
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, TypeVar

from src.db.schemas import CourierModel
from src.model_manager.vllm_model_pool import vLLMModelPool, registry

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

    async def get_model_entry(self, model_name: str) -> Optional[ModelEntry]:
        """Get model entry if exists and not expired"""
        entry = self.models.get(model_name)
        if entry:
            if not entry.is_expired:
                entry.mark_used()
                return entry
            else:
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
        Unified async inference method.
        """
        pool = await self.get_pool(model)
        if not pool:
            return {"error": "Failed to load model or get pool", "status_code": 500}

        try:
            result = await pool.infer(payload)
            return result
        except Exception as e:
            return {"error": f"Inference error: {str(e)}", "status_code": 500}

    async def ensure_model_loaded(self, model: CourierModel) -> Optional[ModelEntry]:
        """Ensure model is loaded, handle memory management"""
        model_name = model.name

        existing = await self.get_model_entry(model_name)
        if existing:
            return existing

        required_gb = model.weights_gb
        available_gb = self.get_available_memory()
        print(f"Available memory: {available_gb:.2f} GB, Required: {required_gb:.2f} GB")

        if available_gb >= (required_gb * 1.1):
            return await self._load_model(model)
        else:
            return await self._load_with_memory_management(model, available_gb, required_gb)

    async def _load_model(self, model: CourierModel) -> Optional[ModelEntry]:
        """Load model into memory"""
        try:
            pool = registry.ensure_pool(model)

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
            return entry
        except Exception as e:
            print(f"Failed to load model {model.name}: {e}")
            return None

    async def _load_with_memory_management(self, model: CourierModel, available_gb: float, required_gb: float) -> Optional[ModelEntry]:
        """Handle memory management when OOM would occur"""
        models_to_unload = self._find_models_to_unload(required_gb, available_gb)

        for model_name in models_to_unload:
            await self._unload_model(model_name)

        new_available_gb = self.get_available_memory()
        if new_available_gb >= required_gb:
            return await self._load_model(model)
        else:
            print(f"Insufficient memory: {required_gb}GB needed, {new_available_gb}GB available")
            return None

    def _find_models_to_unload(self, required_gb: float, available_gb: float) -> List[str]:
        """Find models to unload using LRU policy"""
        candidate_entries = [e for e in self.models.values() if not e.is_expired]
        candidate_entries.sort(key=lambda x: x.last_used)

        total_free = available_gb
        models_to_unload = []

        if total_free < required_gb:
            for entry in candidate_entries:
                if total_free >= required_gb:
                    break
                models_to_unload.append(entry.model.name)
                total_free += entry.size_in_gb

        return models_to_unload

    async def _unload_model(self, model_name: str):
        """Unload a model from memory"""
        if model_name in self.models:
            entry = self.models[model_name]
            await registry.remove_pool(entry.model.model_id)
            del self.models[model_name]
            self.loaded_models = [name for name in self.loaded_models if name != model_name]
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Unloaded model: {model_name}")

    def get_available_memory(self) -> float:
        """Get available memory in GB"""
        if torch.cuda.is_available():
            # Use torch.cuda to get free memory
            free_mem, total_mem = torch.cuda.mem_get_info()
            return free_mem / (1024 ** 3)
        else:
            return psutil.virtual_memory().available / (1024 ** 3)

    async def cleanup_expired_models(self):
        """Clean up all expired models"""
        expired = [name for name, entry in self.models.items() if entry.is_expired]
        for model_name in expired:
            await self._unload_model(model_name)


# Global model manager instance
model_manager = ModelManager()
