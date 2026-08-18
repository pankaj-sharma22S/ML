"""Evidence-driven decision engine evaluating state, gaps, and routing."""

import uuid
from typing import List
from amea.core.capabilities import Capability, CapabilityRegistry
from amea.core.state import (
    GlobalState,
    LifecyclePhase,
    OrchestratorDecision,
    GapItem,
    GapSeverity,
)


class DecisionEngine:
    """Evaluates GlobalState to produce auditable, structured routing decisions."""

    def __init__(self, capability_registry: CapabilityRegistry):
        self.registry = capability_registry

    def evaluate(self, state: GlobalState) -> OrchestratorDecision:
        """Analyze current state and formulate structured decision."""
        phase = state.current_phase
        known_facts: List[str] = []
        missing_info: List[str] = []
        gaps: List[GapItem] = []
        required_caps: List[str] = []
        selected_agents: List[str] = []
        rejected_agents: List[str] = []
        parallel_groups: List[List[str]] = []
        confidence = 1.0
        rationale = ""
        next_action = ""

        # Fact discovery
        if state.user_request:
            known_facts.append(f"User Request: {state.user_request[:80]}...")
        if state.task_spec:
            known_facts.append(f"Task Type: {state.task_spec.task_type.value}, Target: {state.task_spec.target_column}")
        if state.data_profile:
            known_facts.append(f"Data Profile: {state.data_profile.total_rows} rows, {state.data_profile.total_columns} cols")
        if state.experiment_ledger:
            known_facts.append(f"Completed Experiments: {len(state.experiment_ledger)}")

        # Phase-specific reasoning
        if phase == LifecyclePhase.UNDERSTAND:
            if not state.task_spec:
                missing_info.append("Formal ML Task Specification")
                gaps.append(GapItem(
                    description="Task specification not yet compiled from user request",
                    severity=GapSeverity.CRITICAL,
                    affected_components=["ProblemUnderstandingAgent"]
                ))
            required_caps = [Capability.PROBLEM_UNDERSTANDING.value]
            rationale = "Task specification must be synthesized and formalized before data inspection."
            next_action = "Execute ProblemUnderstandingAgent to define MLTaskSpecification"

        elif phase == LifecyclePhase.INSPECT:
            if not state.data_profile:
                missing_info.append("Dataset Statistical Profile")
                gaps.append(GapItem(
                    description="Raw dataset has not been profiled",
                    severity=GapSeverity.CRITICAL,
                    affected_components=["DataProfiler"]
                ))
            required_caps = [Capability.DATA_PROFILING.value]
            rationale = "Dataset must be statistically profiled and checked for basic properties."
            next_action = "Execute DataProfiler to build DataProfile"

        elif phase == LifecyclePhase.VALIDATE:
            required_caps = [Capability.DATA_QUALITY.value, Capability.LEAKAGE_DETECTION.value]
            parallel_groups = [[Capability.DATA_QUALITY.value, Capability.LEAKAGE_DETECTION.value]]
            rationale = "Audit data quality and potential leakage risks in parallel before strategy formation."
            next_action = "Run DataQualityGuard and LeakageDetector"

        elif phase == LifecyclePhase.PLAN:
            required_caps = [Capability.ML_STRATEGY.value, Capability.FEATURE_ENGINEERING.value]
            rationale = "Formulate ML strategy and construct experiment queue based on data profile and task spec."
            next_action = "Formulate candidate model experiments and enqueue configurations"

        elif phase == LifecyclePhase.DISPATCH or phase == LifecyclePhase.EXECUTE:
            required_caps = [Capability.EXPERIMENT_RUNNER.value]
            candidate_count = len(state.experiment_queue)
            rationale = f"Dispatch {candidate_count} enqueued experiments to isolated worker sandboxes."
            next_action = "Execute experiment queue across concurrent workers"

        elif phase == LifecyclePhase.OBSERVE:
            required_caps = [Capability.EXPERIMENT_TRACKER.value]
            rationale = "Collect all completed run records and verify metric integrity."
            next_action = "Commit run results to experiment ledger"

        elif phase == LifecyclePhase.EVALUATE:
            required_caps = [Capability.EVALUATION.value, Capability.JUDGE.value]
            rationale = "Audit models for overfitting/leakage and select Pareto-optimal best candidate."
            next_action = "Run EvaluationAgent and JudgeAgent"

        elif phase == LifecyclePhase.REPLAN:
            required_caps = [Capability.IMPROVEMENT_PLANNER.value]
            rationale = "Performance criteria unmet and budget remains. Formulate delta experiment hypotheses."
            next_action = "Generate targeted improvement experiments without repeating past failures"

        elif phase == LifecyclePhase.BUILD:
            required_caps = [Capability.CODE_GENERATION.value]
            rationale = "Synthesize modular production Python code for the winning validated model."
            next_action = "Generate data_loader.py, features.py, train.py, and inference.py"

        elif phase == LifecyclePhase.VERIFY:
            required_caps = [Capability.CODE_EXECUTION.value, Capability.CODE_REPAIR.value]
            rationale = "Execute generated pipeline code in target sandbox to verify execution and metrics."
            next_action = "Test synthesized pipeline; invoke CodeRepairLoop if errors occur"

        elif phase == LifecyclePhase.FINALIZE:
            required_caps = []
            rationale = "All validation requirements satisfied. Compile final ML engineering report."
            next_action = "Emit FinalReport and terminate workflow"

        # Resolve providers for required capabilities
        for cap_name in required_caps:
            try:
                cap_enum = Capability(cap_name)
                providers = self.registry.get_providers_for_capability(cap_enum)
                if providers:
                    selected_agents.append(providers[0].name)
                    # Remaining providers marked as rejected/alternative
                    for alt in providers[1:]:
                        rejected_agents.append(alt.name)
                else:
                    gaps.append(GapItem(
                        description=f"No active provider registered for capability: {cap_name}",
                        severity=GapSeverity.CRITICAL,
                        affected_components=[cap_name]
                    ))
                    confidence = 0.5
            except ValueError:
                pass

        return OrchestratorDecision(
            decision_id=f"dec_{phase.value}_{uuid.uuid4().hex[:8]}",
            phase=phase,
            objective=state.user_request[:120] if state.user_request else "Default ML Objective",
            known_facts=known_facts,
            missing_information=missing_info,
            identified_gaps=gaps,
            required_capabilities=required_caps,
            selected_agents=list(set(selected_agents)),
            rejected_agents=list(set(rejected_agents)),
            parallel_groups=parallel_groups,
            confidence_score=confidence,
            rationale=rationale,
            next_action=next_action,
        )
