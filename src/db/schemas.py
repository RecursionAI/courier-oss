# Pydantic models for data validation and FlowDB persistence.

from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Any


class CourierMembership(BaseModel):
    id: str
    api_key: str
    role: str


# `CourierModel`: Defines the configuration for a model, including its file path, 
# VRAM requirements (`weights_gb`), and `api_type`.
# - "static": Models that stay in memory permanently.
# - "flex": Models that are subject to the TTL/LRU unloading logic.
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
