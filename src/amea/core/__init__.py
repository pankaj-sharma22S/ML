"""Core abstractions, state models, events, and configuration."""

from amea.core.exceptions import (
    AMEAException,
    StateValidationError,
    InvalidTransitionError,
    BudgetExceededError,
    SecurityViolationError,
    ExecutionTimeoutError,
    ResourceConstraintError,
)
from amea.core.config import ProjectConfig, ComputeBudget, ExecutionLimits
from amea.core.events import EventBus, TelemetryEvent, EventType
from amea.core.state import GlobalState, LifecyclePhase, MLTaskSpecification
from amea.core.capabilities import Capability, CapabilityRegistry

__all__ = [
    "AMEAException",
    "StateValidationError",
    "InvalidTransitionError",
    "BudgetExceededError",
    "SecurityViolationError",
    "ExecutionTimeoutError",
    "ResourceConstraintError",
    "ProjectConfig",
    "ComputeBudget",
    "ExecutionLimits",
    "EventBus",
    "TelemetryEvent",
    "EventType",
    "GlobalState",
    "LifecyclePhase",
    "MLTaskSpecification",
    "Capability",
    "CapabilityRegistry",
]
