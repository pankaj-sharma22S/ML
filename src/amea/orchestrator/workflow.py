"""Workflow StateGraph engine coordinating lifecycle nodes and telemetry."""

from typing import Callable, Dict
from amea.core.config import ComputeBudget
from amea.core.events import EventBus, EventType
from amea.core.exceptions import BudgetExceededError
from amea.core.state import GlobalState, LifecyclePhase
from amea.orchestrator.decision_engine import DecisionEngine
from amea.orchestrator.nodes import OrchestratorNodes
from amea.persistence.checkpointer import StateCheckpointer


class OrchestratorWorkflow:
    """Deterministic, framework-independent workflow coordinating the ML lifecycle."""

    def __init__(
        self,
        nodes: OrchestratorNodes,
        decision_engine: DecisionEngine,
        event_bus: EventBus,
        checkpointer: StateCheckpointer,
        budget: ComputeBudget,
    ):
        self.nodes = nodes
        self.decision_engine = decision_engine
        self.event_bus = event_bus
        self.checkpointer = checkpointer
        self.budget = budget

        # Node map by phase
        self._node_dispatch: Dict[LifecyclePhase, Callable[[GlobalState], GlobalState]] = {
            LifecyclePhase.UNDERSTAND: self.nodes.understand_node,
            LifecyclePhase.INSPECT: self.nodes.inspect_node,
            LifecyclePhase.VALIDATE: self.nodes.validate_node,
            LifecyclePhase.PLAN: self.nodes.plan_node,
            LifecyclePhase.DISPATCH: self.nodes.dispatch_and_execute_node,
            LifecyclePhase.EXECUTE: self.nodes.dispatch_and_execute_node,
            LifecyclePhase.EVALUATE: self.nodes.evaluate_node,
            LifecyclePhase.BUILD: self.nodes.build_node,
            LifecyclePhase.VERIFY: self.nodes.verify_node,
            LifecyclePhase.FINALIZE: self.nodes.finalize_node,
        }

    def step(self, state: GlobalState) -> GlobalState:
        """Execute a single lifecycle step, emitting decision, executing node, and recording checkpoint."""
        if state.is_terminal or state.current_phase == LifecyclePhase.TERMINATED:
            return state

        current_phase = state.current_phase

        # 1. Budget enforcement check
        if len(state.experiment_ledger) >= self.budget.max_experiments:
            self.event_bus.publish(
                EventType.SECURITY_BLOCKED,
                "Orchestrator",
                f"Experiment budget of {self.budget.max_experiments} reached. Transitioning to evaluation/finalization.",
            )

        # 2. Decision Engine evaluation
        decision = self.decision_engine.evaluate(state)
        self.event_bus.publish(
            EventType.DECISION_CREATED,
            "DecisionEngine",
            f"Decision for phase {current_phase}: {decision.next_action}",
            {"decision_id": decision.decision_id, "rationale": decision.rationale},
        )

        # 3. Node Dispatch
        node_fn = self._node_dispatch.get(current_phase)
        if not node_fn:
            # Unmapped phase -> terminate safely
            state.is_terminal = True
            state.termination_reason = f"No node dispatch registered for phase {current_phase}"
            state.current_phase = LifecyclePhase.TERMINATED
            return state

        self.event_bus.publish(
            EventType.TASK_STARTED,
            "Orchestrator",
            f"Starting lifecycle node: {current_phase.value}",
        )

        # Execute node
        next_state = node_fn(state)

        self.event_bus.publish(
            EventType.TASK_COMPLETED,
            "Orchestrator",
            f"Completed lifecycle node: {current_phase.value} -> next phase: {next_state.current_phase.value}",
        )

        # 4. Checkpoint State
        cp_meta = self.checkpointer.save_checkpoint(next_state)
        self.event_bus.publish(
            EventType.CHECKPOINT_CREATED,
            "StateCheckpointer",
            f"Checkpoint saved: {cp_meta.checkpoint_id}",
            {"checkpoint_id": cp_meta.checkpoint_id, "file_name": cp_meta.file_name},
        )

        return next_state

    def run(self, initial_state: GlobalState, max_steps: int = 20) -> GlobalState:
        """Run workflow until terminal state or step limit reached."""
        state = initial_state
        step_count = 0

        while not state.is_terminal and step_count < max_steps:
            state = self.step(state)
            step_count += 1

        if step_count >= max_steps and not state.is_terminal:
            state.is_terminal = True
            state.termination_reason = "Max workflow execution steps reached without termination"
            state.current_phase = LifecyclePhase.TERMINATED

        return state
