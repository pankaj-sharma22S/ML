"""Observability, request tracing, and latency tracking for RAG pipelines."""

import time
from typing import Any, Dict, List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class StageTrace(BaseModel):
    stage_name: str
    duration_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RequestTrace(BaseModel):
    request_id: str
    query: str
    tenant_id: str
    total_duration_ms: float
    stages: List[StageTrace] = Field(default_factory=list)
    cache_hit: bool = False
    conflicts_count: int = 0


class RAGObservabilityTracer:
    """Collects and summarizes telemetry across RAG pipeline components."""

    def __init__(self):
        self.traces: List[RequestTrace] = []

    def start_trace(self, query: str, tenant_id: str = "default_tenant") -> str:
        req_id = f"trace_{uuid4().hex[:8]}"
        return req_id

    def record_trace(
        self,
        request_id: str,
        query: str,
        tenant_id: str,
        total_duration_ms: float,
        stages: List[StageTrace],
        cache_hit: bool = False,
        conflicts_count: int = 0,
    ) -> RequestTrace:
        trace = RequestTrace(
            request_id=request_id,
            query=query,
            tenant_id=tenant_id,
            total_duration_ms=total_duration_ms,
            stages=stages,
            cache_hit=cache_hit,
            conflicts_count=conflicts_count,
        )
        self.traces.append(trace)
        return trace
