from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Any


class CourierMembership(BaseModel):
    id: str
    api_key: str
    role: str


class CourierModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    nickname: Optional[str]
    model_id: str
    name: str
    # modules: List[Module]
    memberships: List[CourierMembership]
    family: str
    context_window: int
    adapter_path: Optional[str]
    model_type: str
    file_path: str
    api_type: str  # flex, static
    weights_gb: float
    instances: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 16000
    max_num_seqs: int = 2


class CourierUser(BaseModel):
    id: str
    api_key: str
    valid: bool


class AnalyticsRequest(BaseModel):
    model_name: str
    prompt_tokens: int
    generation_tokens: int
    start_time: str
    end_time: str
    peak_memory: float
    system_active_memory: str = "16.0 GB"


class Analytics(BaseModel):
    id: str
    requests: List[AnalyticsRequest]


class TrendRequest(BaseModel):
    api_key: str
    limit: int
    skip: int
