"""Auditable event bus and telemetry definitions."""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List
from pydantic import BaseModel, Field


class EventType(str, Enum):
    TASK_CREATED = "TASK_CREATED"
    TASK_STARTED = "TASK_STARTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    AGENT_DISPATCHED = "AGENT_DISPATCHED"
    EXPERIMENT_STARTED = "EXPERIMENT_STARTED"
    EXPERIMENT_COMPLETED = "EXPERIMENT_COMPLETED"
    DECISION_CREATED = "DECISION_CREATED"
    PLAN_UPDATED = "PLAN_UPDATED"
    ARTIFACT_CREATED = "ARTIFACT_CREATED"
    SECURITY_BLOCKED = "SECURITY_BLOCKED"
    CHECKPOINT_CREATED = "CHECKPOINT_CREATED"
    STATE_TRANSITION = "STATE_TRANSITION"


class TelemetryEvent(BaseModel):
    """Immutable auditable event record."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = Field(default="trace-default")
    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: EventType
    source_component: str
    message: str
    attributes: Dict[str, Any] = Field(default_factory=dict)


class EventBus:
    """Thread-safe event dispatcher and ledger."""

    def __init__(self, trace_id: str = "trace-default"):
        self.trace_id = trace_id
        self._subscribers: Dict[EventType, List[Callable[[TelemetryEvent], None]]] = {}
        self._global_subscribers: List[Callable[[TelemetryEvent], None]] = []
        self._history: List[TelemetryEvent] = []

    def subscribe(self, event_type: EventType, callback: Callable[[TelemetryEvent], None]) -> None:
        """Register a callback for a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def subscribe_all(self, callback: Callable[[TelemetryEvent], None]) -> None:
        """Register a callback for all events."""
        self._global_subscribers.append(callback)

    def publish(
        self,
        event_type: EventType,
        source_component: str,
        message: str,
        attributes: Dict[str, Any] | None = None,
    ) -> TelemetryEvent:
        """Publish a new telemetry event to subscribers and append to ledger."""
        event = TelemetryEvent(
            trace_id=self.trace_id,
            event_type=event_type,
            source_component=source_component,
            message=message,
            attributes=attributes or {},
        )
        self._history.append(event)

        # Notify specific subscribers
        for callback in self._subscribers.get(event_type, []):
            try:
                callback(event)
            except Exception:
                pass  # Event bus callbacks must not break execution

        # Notify global subscribers
        for callback in self._global_subscribers:
            try:
                callback(event)
            except Exception:
                pass

        return event

    def get_history(self, event_type: EventType | None = None) -> List[TelemetryEvent]:
        """Retrieve full or filtered event ledger."""
        if event_type is None:
            return list(self._history)
        return [e for e in self._history if e.event_type == event_type]
