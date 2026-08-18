"""High-level Orchestrator Runner coordinating end-to-end execution."""

from pathlib import Path
from typing import Optional, Dict, Any

from amea.core.capabilities import Capability, CapabilityProvider, CapabilityRegistry
from amea.core.config import ProjectConfig, ComputeBudget
from amea.core.events import EventBus
from amea.core.state import GlobalState, LifecyclePhase
from amea.execution.subprocess_executor import SubprocessExecutor
from amea.orchestrator.decision_engine import DecisionEngine
from amea.orchestrator.nodes import OrchestratorNodes
from amea.orchestrator.workflow import OrchestratorWorkflow
from amea.persistence.checkpointer import StateCheckpointer
from amea.tools.system_inspector import SystemInspector


class OrchestratorRunner:
    """Entry point for executing an Autonomous ML Engineer project run."""

    def __init__(self, config: Optional[ProjectConfig] = None):
        self.config = config or ProjectConfig()
        self.event_bus = EventBus(trace_id=f"run_{self.config.project_id}")
        self.registry = CapabilityRegistry()
        self.inspector = SystemInspector()
        self.checkpointer = StateCheckpointer(persistence_dir=self.config.persistence.project_dir)
        self.executor = SubprocessExecutor(sandbox_root=self.config.security.sandbox_root)

        self._register_default_capabilities()

    def _register_default_capabilities(self) -> None:
        """Register default core capability providers."""
        providers = [
            CapabilityProvider(name="ProblemUnderstandingAgent", capabilities={Capability.PROBLEM_UNDERSTANDING}, priority=10),
            CapabilityProvider(name="DataProfiler", capabilities={Capability.DATA_PROFILING}, priority=10),
            CapabilityProvider(name="DataQualityGuard", capabilities={Capability.DATA_QUALITY, Capability.LEAKAGE_DETECTION}, priority=10),
            CapabilityProvider(name="MLStrategist", capabilities={Capability.ML_STRATEGY, Capability.FEATURE_ENGINEERING}, priority=10),
            CapabilityProvider(name="ExperimentRunner", capabilities={Capability.EXPERIMENT_RUNNER}, priority=10),
            CapabilityProvider(name="ExperimentTracker", capabilities={Capability.EXPERIMENT_TRACKER}, priority=10),
            CapabilityProvider(name="EvaluationAgent", capabilities={Capability.EVALUATION}, priority=10),
            CapabilityProvider(name="JudgeAgent", capabilities={Capability.JUDGE}, priority=10),
            CapabilityProvider(name="ImprovementPlanner", capabilities={Capability.IMPROVEMENT_PLANNER}, priority=10),
            CapabilityProvider(name="CodeGenerator", capabilities={Capability.CODE_GENERATION}, priority=10),
            CapabilityProvider(name="CodeExecutor", capabilities={Capability.CODE_EXECUTION, Capability.CODE_REPAIR}, priority=10),
        ]
        for p in providers:
            self.registry.register(p)

    def run_task(
        self,
        user_request: str,
        dataset_path: Optional[str] = None,
        target_column: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GlobalState:
        """Execute a full autonomous ML task from request to verified pipeline."""
        # 1. System capability inspection (factual hardware discovery)
        hw_capabilities = self.inspector.inspect()

        # 2. Initial State Assembly
        meta = metadata or {}
        if dataset_path:
            meta["dataset_path"] = dataset_path
        if target_column:
            meta["target_column"] = target_column

        initial_state = GlobalState(
            project_id=self.config.project_id,
            task_id="task-001",
            user_request=user_request,
            current_phase=LifecyclePhase.UNDERSTAND,
            dataset_metadata=meta,
        )

        # 3. Assemble Workflow
        decision_engine = DecisionEngine(capability_registry=self.registry)
        nodes = OrchestratorNodes(executor=self.executor)
        workflow = OrchestratorWorkflow(
            nodes=nodes,
            decision_engine=decision_engine,
            event_bus=self.event_bus,
            checkpointer=self.checkpointer,
            budget=self.config.budget,
        )

        # 4. Run StateGraph
        final_state = workflow.run(initial_state)
        return final_state
