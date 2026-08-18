"""Validated state reducers with strict state ownership rules."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from amea.core.exceptions import StateOwnershipError, InvalidTransitionError
from amea.core.state import (
    GlobalState,
    LifecyclePhase,
    MLTaskSpecification,
    DataProfile,
    ExperimentConfiguration,
    RegisteredExperimentRecord,
    EvaluationAuditReport,
    JudgeDecision,
    GeneratedCodeArtifacts,
    PipelineExecutionResult,
    FinalReport,
    OrchestratorDecision,
    GapItem,
)


class StatePatch(BaseModel):
    """Encapsulates a proposed state delta signed by the updating component."""
    author_component: str
    target_phase: Optional[LifecyclePhase] = None
    task_spec: Optional[MLTaskSpecification] = None
    dataset_metadata: Optional[Dict[str, Any]] = None
    data_profile: Optional[DataProfile] = None
    data_quality_report: Optional[Dict[str, Any]] = None
    eda_findings: Optional[List[str]] = None
    new_experiments: Optional[List[ExperimentConfiguration]] = None
    new_experiment_records: Optional[List[RegisteredExperimentRecord]] = None
    new_audit_reports: Optional[Dict[str, EvaluationAuditReport]] = None
    best_candidate: Optional[RegisteredExperimentRecord] = None
    judge_decision: Optional[JudgeDecision] = None
    code_artifacts: Optional[GeneratedCodeArtifacts] = None
    execution_result: Optional[PipelineExecutionResult] = None
    final_report: Optional[FinalReport] = None
    new_decision: Optional[OrchestratorDecision] = None
    new_gaps: Optional[List[GapItem]] = None
    new_assumptions: Optional[List[str]] = None
    task_started: Optional[str] = None
    task_completed: Optional[str] = None
    task_failed: Optional[str] = None
    increment_iteration: bool = False
    termination_reason: Optional[str] = None
    is_terminal: Optional[bool] = None


# Valid state transitions matrix
VALID_TRANSITIONS: Dict[LifecyclePhase, List[LifecyclePhase]] = {
    LifecyclePhase.UNDERSTAND: [LifecyclePhase.INSPECT, LifecyclePhase.TERMINATED],
    LifecyclePhase.INSPECT: [LifecyclePhase.VALIDATE, LifecyclePhase.TERMINATED],
    LifecyclePhase.VALIDATE: [LifecyclePhase.PLAN, LifecyclePhase.UNDERSTAND, LifecyclePhase.TERMINATED],
    LifecyclePhase.PLAN: [LifecyclePhase.DISPATCH, LifecyclePhase.EXECUTE, LifecyclePhase.TERMINATED],
    LifecyclePhase.DISPATCH: [LifecyclePhase.EXECUTE, LifecyclePhase.OBSERVE, LifecyclePhase.EVALUATE, LifecyclePhase.TERMINATED],
    LifecyclePhase.EXECUTE: [LifecyclePhase.OBSERVE, LifecyclePhase.EVALUATE, LifecyclePhase.TERMINATED],
    LifecyclePhase.OBSERVE: [LifecyclePhase.EVALUATE, LifecyclePhase.TERMINATED],
    LifecyclePhase.EVALUATE: [LifecyclePhase.BUILD, LifecyclePhase.REPLAN, LifecyclePhase.TERMINATED],
    LifecyclePhase.REPLAN: [LifecyclePhase.PLAN, LifecyclePhase.BUILD, LifecyclePhase.TERMINATED],
    LifecyclePhase.BUILD: [LifecyclePhase.VERIFY, LifecyclePhase.TERMINATED],
    LifecyclePhase.VERIFY: [LifecyclePhase.FINALIZE, LifecyclePhase.BUILD, LifecyclePhase.TERMINATED],
    LifecyclePhase.FINALIZE: [LifecyclePhase.TERMINATED],
    LifecyclePhase.TERMINATED: [],
}

# Strict ownership of state fields
COMPONENT_PERMISSIONS: Dict[str, List[str]] = {
    "Orchestrator": ["target_phase", "new_decision", "task_started", "task_completed", "task_failed", "increment_iteration", "termination_reason", "is_terminal"],
    "ProblemUnderstandingAgent": ["task_spec", "new_gaps", "new_assumptions", "target_phase"],
    "DataProfiler": ["data_profile", "dataset_metadata", "data_quality_report", "eda_findings", "target_phase"],
    "DataQualityGuard": ["data_quality_report", "new_gaps", "target_phase"],
    "DataCleaningAgent": ["data_profile", "dataset_metadata", "target_phase"],
    "DataValidationAgent": ["data_quality_report", "new_gaps", "target_phase"],
    "EDAAgent": ["eda_findings", "data_quality_report", "new_assumptions", "target_phase"],
    "MLStrategist": ["new_experiments", "new_assumptions", "target_phase"],
    "ExperimentRunner": ["new_experiment_records", "task_completed", "task_failed", "target_phase"],
    "EvaluationAgent": ["new_audit_reports", "target_phase"],
    "JudgeAgent": ["best_candidate", "judge_decision", "new_audit_reports", "target_phase"],
    "ImprovementPlanner": ["new_experiments", "new_gaps", "target_phase"],
    "CodeGenerator": ["code_artifacts", "target_phase"],
    "CodeExecutor": ["execution_result", "target_phase", "termination_reason", "is_terminal"],
    "CodeRepairLoop": ["code_artifacts", "execution_result", "target_phase"],
    "Finalizer": ["final_report", "termination_reason", "is_terminal", "target_phase"],
}


def apply_state_patch(current_state: GlobalState, patch: StatePatch) -> GlobalState:
    """Apply a validated state patch and return a new updated GlobalState."""
    author = patch.author_component
    allowed_fields = COMPONENT_PERMISSIONS.get(author, [])

    # Create shallow copy for mutation
    state_dict = current_state.model_dump()

    # Validate and apply phase transition
    if patch.target_phase is not None:
        if "target_phase" not in allowed_fields and author != "Orchestrator":
            raise StateOwnershipError(f"Component '{author}' is not authorized to transition lifecycle phase.")
        allowed_next = VALID_TRANSITIONS.get(current_state.current_phase, [])
        if patch.target_phase not in allowed_next and patch.target_phase != LifecyclePhase.TERMINATED:
            raise InvalidTransitionError(
                f"Invalid phase transition: {current_state.current_phase} -> {patch.target_phase}. Allowed: {allowed_next}"
            )
        state_dict["current_phase"] = patch.target_phase

    # Field-by-field authorized update
    if patch.task_spec is not None:
        _check_permission(author, "task_spec", allowed_fields)
        state_dict["task_spec"] = patch.task_spec.model_dump()

    if patch.data_profile is not None:
        _check_permission(author, "data_profile", allowed_fields)
        state_dict["data_profile"] = patch.data_profile.model_dump()

    if patch.dataset_metadata is not None:
        _check_permission(author, "dataset_metadata", allowed_fields)
        state_dict["dataset_metadata"].update(patch.dataset_metadata)

    if patch.data_quality_report is not None:
        _check_permission(author, "data_quality_report", allowed_fields)
        state_dict["data_quality_report"].update(patch.data_quality_report)

    if patch.eda_findings is not None:
        _check_permission(author, "eda_findings", allowed_fields)
        state_dict["eda_findings"].extend(patch.eda_findings)

    if patch.new_experiments is not None:
        _check_permission(author, "new_experiments", allowed_fields)
        state_dict["experiment_queue"].extend([e.model_dump() for e in patch.new_experiments])

    if patch.new_experiment_records is not None:
        _check_permission(author, "new_experiment_records", allowed_fields)
        state_dict["experiment_ledger"].extend([r.model_dump() for r in patch.new_experiment_records])

    if patch.new_audit_reports is not None:
        _check_permission(author, "new_audit_reports", allowed_fields)
        for k, v in patch.new_audit_reports.items():
            state_dict["audit_reports"][k] = v.model_dump()

    if patch.best_candidate is not None:
        _check_permission(author, "best_candidate", allowed_fields)
        state_dict["best_candidate"] = patch.best_candidate.model_dump()

    if patch.judge_decision is not None:
        _check_permission(author, "judge_decision", allowed_fields)
        state_dict["judge_decision"] = patch.judge_decision.model_dump()

    if patch.code_artifacts is not None:
        _check_permission(author, "code_artifacts", allowed_fields)
        state_dict["code_artifacts"] = patch.code_artifacts.model_dump()

    if patch.execution_result is not None:
        _check_permission(author, "execution_result", allowed_fields)
        state_dict["execution_result"] = patch.execution_result.model_dump()

    if patch.final_report is not None:
        _check_permission(author, "final_report", allowed_fields)
        state_dict["final_report"] = patch.final_report.model_dump()

    if patch.new_decision is not None:
        _check_permission(author, "new_decision", allowed_fields)
        state_dict["decisions_history"].append(patch.new_decision.model_dump())

    if patch.new_gaps is not None:
        _check_permission(author, "new_gaps", allowed_fields)
        state_dict["identified_gaps"].extend([g.model_dump() for g in patch.new_gaps])

    if patch.new_assumptions is not None:
        _check_permission(author, "new_assumptions", allowed_fields)
        state_dict["assumptions"].extend(patch.new_assumptions)

    if patch.task_started is not None:
        if patch.task_started not in state_dict["active_tasks"]:
            state_dict["active_tasks"].append(patch.task_started)

    if patch.task_completed is not None:
        if patch.task_completed in state_dict["active_tasks"]:
            state_dict["active_tasks"].remove(patch.task_completed)
        if patch.task_completed not in state_dict["completed_tasks"]:
            state_dict["completed_tasks"].append(patch.task_completed)

    if patch.task_failed is not None:
        if patch.task_failed in state_dict["active_tasks"]:
            state_dict["active_tasks"].remove(patch.task_failed)
        if patch.task_failed not in state_dict["failed_tasks"]:
            state_dict["failed_tasks"].append(patch.task_failed)

    if patch.increment_iteration:
        state_dict["iteration"] += 1

    if patch.termination_reason is not None:
        state_dict["termination_reason"] = patch.termination_reason

    if patch.is_terminal is not None:
        state_dict["is_terminal"] = patch.is_terminal

    return GlobalState.model_validate(state_dict)


def _check_permission(author: str, field_name: str, allowed_fields: List[str]) -> None:
    if field_name not in allowed_fields and author != "Orchestrator":
        raise StateOwnershipError(f"Component '{author}' is not authorized to write to state field '{field_name}'.")
