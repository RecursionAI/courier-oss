from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Any, Dict
import uuid


class Model(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_id: uuid.UUID
    name: str
    models: List[Any]
    # tokenizer: TokenizerWrapper
    tokenizer: Any
    context_window: int
    adapter_path: Optional[str]
    model_type: str
    api_key: uuid.UUID


class VisionModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_id: uuid.UUID
    name: str
    models: List[Any]
    tokenizer: Any
    context_window: int
    adapter_path: Optional[str]
    model_type: str
    api_key: uuid.UUID


class ModelRequest(BaseModel):
    model_name: str
    context_window: int
    adapter_path: Optional[str]
    api_key: uuid.UUID


class DeleteModelRequest(BaseModel):
    name: str
    model_id: uuid.UUID
    api_key: uuid.UUID

class DeleteLibModelRequest(BaseModel):
    name: str
    api_key: str


class RemoveModelRequest(BaseModel):
    model_name: str
    model_id: uuid.UUID
    company_id: uuid.UUID


class NewModelRequest(BaseModel):
    name: str
    nickname: Optional[str]
    context_window: int
    adapter_path: Optional[str]
    api_type: str
    api_key: uuid.UUID
    instances: int = 1


class NewLibModelRequest(BaseModel):
    api_key: str
    name: str
    family: str
    context_window: int
    adapter_path: Optional[str]
    model_type: str
    file_path: str
    weights_gb: float


# from pydantic import BaseModel, Field
# from uuid import UUID
# from typing import Optional

class InferenceRequest(BaseModel):
    model_name: str
    model_id: uuid.UUID
    model_type: str
    api_key: uuid.UUID
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    sampling_params: Optional[Dict[str, Any]] = None
    stream: bool = False


class ReplaceUserRequest(BaseModel):
    previous_id: uuid.UUID
    new_id: uuid.UUID


class UpdateAdapterRequest(BaseModel):
    model_name: str
    model_id: uuid.UUID
    adapter_path: str
    context_window: int
    api_key: uuid.UUID


class DeleteAdapterRequest(BaseModel):
    adapter_path: str
    dataset_id: uuid.UUID


class VoxtralModelRequest(BaseModel):
    model_name: str
    company_name: str
    company_id: uuid.UUID
    context_window: int
    adapter_path: Optional[str] = None
    instances: Optional[int] = None
