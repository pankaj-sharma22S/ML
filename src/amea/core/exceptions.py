"""Domain exception hierarchy for Autonomous ML Engineer Agent."""

class AMEAException(Exception):
    """Base exception for all AMEA domain errors."""
    pass


class StateValidationError(AMEAException):
    """Raised when state mutation violates schema or invariant rules."""
    pass


class InvalidTransitionError(AMEAException):
    """Raised when an illegal lifecycle phase transition is attempted."""
    pass


class StateOwnershipError(AMEAException):
    """Raised when a component attempts to modify state it does not own."""
    pass


class BudgetExceededError(AMEAException):
    """Raised when compute, time, or experiment limits are breached."""
    pass


class SecurityViolationError(AMEAException):
    """Raised when code execution or file access violates security boundaries."""
    pass


class ExecutionTimeoutError(AMEAException):
    """Raised when a task or sandbox worker exceeds allotted duration."""
    pass


class ResourceConstraintError(AMEAException):
    """Raised when requested resources exceed available or authorized limits."""
    pass


class CheckpointRecoveryError(AMEAException):
    """Raised when checkpoint loading or state deserialization fails."""
    pass


class TaskDependencyError(AMEAException):
    """Raised when task graph dependencies are cyclical or unsatisfiable."""
    pass
