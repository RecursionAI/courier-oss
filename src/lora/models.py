from typing import Optional

from pydantic import BaseModel
import uuid


class LoraRequest(BaseModel):
    model_name: str
    display_name: str
    dataset_name: uuid.UUID
    api_key: uuid.UUID
    adapter_path: Optional[str]
    val_size: float


class EvalRequest(BaseModel):
    model_name: str
    adapter_path: str
    max_seq_length: int
    dataset_id: uuid.UUID
    val_size: float
