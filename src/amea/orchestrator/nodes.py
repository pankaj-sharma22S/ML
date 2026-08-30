"""Lifecycle node implementations emitting validated StatePatch objects."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from amea.core.exceptions import BudgetExceededError, SecurityViolationError
from amea.core.reducers import StatePatch, apply_state_patch
from amea.core.state import (
    GlobalState,
    LifecyclePhase,
    MLTaskSpecification,
    TaskType,
    DataProfile,
    ColumnProfile,
    ExperimentConfiguration,
    RegisteredExperimentRecord,
    EvaluationAuditReport,
    JudgeDecision,
    GeneratedCodeArtifacts,
    PipelineExecutionResult,
    FinalReport,
)
from amea.execution.subprocess_executor import SubprocessExecutor


class OrchestratorNodes:
    """Implements lifecycle nodes for the Orchestrator graph."""

    def __init__(self, executor: SubprocessExecutor | None = None):
        self.executor = executor or SubprocessExecutor(sandbox_root=Path(".amea_sandboxes"))

    def understand_node(self, state: GlobalState) -> GlobalState:
        """Formalize user request into structured MLTaskSpecification using ProblemUnderstandingAgent."""
        from amea.problem_understanding.agent import ProblemUnderstandingAgent
        agent = ProblemUnderstandingAgent()

        target_hint = state.dataset_metadata.get("target_column")
        df_sample = None
        dataset_path_str = state.dataset_metadata.get("dataset_path")
        if dataset_path_str and Path(dataset_path_str).exists():
            try:
                df_sample = pd.read_csv(dataset_path_str, nrows=100)
            except Exception:
                df_sample = None

        report = agent.formulate_problem(
            user_request=state.user_request,
            df=df_sample,
            target_column_hint=target_hint,
        )

        patch = StatePatch(
            author_component="ProblemUnderstandingAgent",
            task_spec=report.task_spec,
            new_gaps=report.identified_gaps,
            new_assumptions=report.assumptions_made,
            target_phase=LifecyclePhase.INSPECT,
        )
        return apply_state_patch(state, patch)

    def inspect_node(self, state: GlobalState) -> GlobalState:
        """Statistically profile raw dataset deterministically."""
        dataset_path_str = state.dataset_metadata.get("dataset_path")
        if not dataset_path_str:
            # Look in workspace for .csv files
            csvs = list(Path(".").glob("*.csv")) + list(Path(".").glob("data/*.csv"))
            if csvs:
                dataset_path_str = str(csvs[0])
            else:
                # Missing dataset
                patch = StatePatch(
                    author_component="Orchestrator",
                    target_phase=LifecyclePhase.TERMINATED,
                    termination_reason="Dataset not found in workspace and no dataset_path provided",
                    is_terminal=True,
                )
                return apply_state_patch(state, patch)

        dataset_path = Path(dataset_path_str)
        if not dataset_path.exists():
            patch = StatePatch(
                author_component="Orchestrator",
                target_phase=LifecyclePhase.TERMINATED,
                termination_reason=f"Dataset path '{dataset_path}' does not exist",
                is_terminal=True,
            )
            return apply_state_patch(state, patch)

        # Use DataIntelligenceAgent for deep deterministic audit
        from amea.data_intelligence.agent import DataIntelligenceAgent
        data_agent = DataIntelligenceAgent()

        target_candidate = state.task_spec.target_column if state.task_spec else None
        is_classification = (state.task_spec.task_type != TaskType.REGRESSION) if state.task_spec else True

        evidence_pkg, data_profile = data_agent.process_dataset(
            dataset_path=dataset_path,
            target_column=target_candidate,
            is_classification=is_classification,
        )

        # Update target column candidate if not set
        if state.task_spec and not state.task_spec.target_column:
            # Default heuristic: last column if not ID / leakage
            df = pd.read_csv(dataset_path)
            candidates = [c for c in df.columns if c not in data_profile.potential_leakage_columns]
            if candidates:
                state.task_spec.target_column = candidates[-1]
                state.task_spec.feature_candidates = [c for c in candidates if c != state.task_spec.target_column]

        patch = StatePatch(
            author_component="DataProfiler",
            data_profile=data_profile,
            data_quality_report=evidence_pkg.quality_audit.model_dump(mode="json"),
            eda_findings=evidence_pkg.summary_findings,
            dataset_metadata={"dataset_version": evidence_pkg.dataset_version.model_dump(mode="json")},
            target_phase=LifecyclePhase.VALIDATE,
        )
        return apply_state_patch(state, patch)

    def validate_node(self, state: GlobalState) -> GlobalState:
        """Validate data quality, target availability, and leakage boundaries."""
        if not state.data_profile or not state.task_spec:
            patch = StatePatch(
                author_component="Orchestrator",
                target_phase=LifecyclePhase.TERMINATED,
                termination_reason="Missing data profile or task specification during validation",
                is_terminal=True,
            )
            return apply_state_patch(state, patch)

        if not state.task_spec.target_column or state.task_spec.target_column not in state.data_profile.columns:
            patch = StatePatch(
                author_component="Orchestrator",
                target_phase=LifecyclePhase.TERMINATED,
                termination_reason=f"Target column '{state.task_spec.target_column}' not found in dataset columns",
                is_terminal=True,
            )
            return apply_state_patch(state, patch)

        # Run EDAAgent for deep distribution, outlier, and target diagnostics
        from amea.eda.agent import EDAAgent
        eda_agent = EDAAgent()
        df = pd.read_csv(state.data_profile.dataset_path)

        is_classification = (state.task_spec.task_type != TaskType.REGRESSION)
        eda_report = eda_agent.analyze(
            df=df,
            dataset_name=Path(state.data_profile.dataset_path).stem,
            target_column=state.task_spec.target_column,
            is_classification=is_classification,
        )

        # Run DataCleaningAgent to produce clean dataset version
        from amea.data_cleaning.agent import DataCleaningAgent
        from amea.data_intelligence.models import DataTreatmentCandidate
        
        treatment_candidates = []
        for col_name, cp in state.data_profile.columns.items():
            if col_name == state.task_spec.target_column:
                continue
            if cp.null_count > 0:
                treatment_candidates.append(
                    DataTreatmentCandidate(
                        strategy_id=f"impute_{col_name}",
                        target_columns=[col_name],
                        treatment_type="imputation",
                        proposed_transformer="AdaptiveImputerTransformer",
                        rationale=f"Impute missing values in {col_name} to guarantee numeric estimator stability",
                    )
                )

        cleaning_agent = DataCleaningAgent()
        cleaned_artifact = cleaning_agent.clean_dataset(
            raw_dataset_path=state.data_profile.dataset_path,
            treatment_candidates=treatment_candidates,
            target_column=state.task_spec.target_column,
        )
        cleaned_df = pd.read_csv(cleaned_artifact.cleaned_dataset_path)

        # Run DataValidationAgent for independent pre-modeling Quality Gate audit
        from amea.data_validation.agent import DataValidationAgent
        from amea.data_validation.models import QualityGateVerdict
        val_agent = DataValidationAgent()

        gate_report = val_agent.evaluate_quality_gate(
            raw_df=df,
            cleaned_df=cleaned_df,
            dataset_name=Path(state.data_profile.dataset_path).stem,
            target_column=state.task_spec.target_column,
        )

        quality_report = {
            "is_clean": (state.data_profile.duplicate_rows == 0 and len(state.data_profile.potential_leakage_columns) < 3),
            "leakage_risk_columns": state.data_profile.potential_leakage_columns,
            "total_null_ratio": sum(c.null_ratio for c in state.data_profile.columns.values()) / max(1, state.data_profile.total_columns),
            "eda_findings_count": len(eda_report.findings),
            "quality_gate_verdict": gate_report.verdict.value,
            "quality_gate_passed": (gate_report.verdict != QualityGateVerdict.REJECTED_BLOCKING),
            "cleaned_dataset_path": cleaned_artifact.cleaned_dataset_path,
        }

        # Convert findings to concise summaries for state
        findings_summaries = [f"[{f.severity.value}] {f.observation}" for f in eda_report.findings]

        if gate_report.verdict == QualityGateVerdict.REJECTED_BLOCKING:
            patch = StatePatch(
                author_component="DataValidationAgent",
                data_quality_report=quality_report,
                target_phase=LifecyclePhase.TERMINATED,
                termination_reason=f"Quality Gate Rejected: {'; '.join(gate_report.blocking_reasons)}",
                is_terminal=True,
            )
            return apply_state_patch(state, patch)

        patch = StatePatch(
            author_component="EDAAgent",
            data_quality_report=quality_report,
            eda_findings=findings_summaries,
            target_phase=LifecyclePhase.PLAN,
        )
        return apply_state_patch(state, patch)

    def plan_node(self, state: GlobalState) -> GlobalState:
        """Formulate candidate model experiment specifications using MLStrategyAgent."""
        from amea.ml_strategy.agent import MLStrategyAgent
        from amea.ml_strategy.models import MLStrategyContext, StrategyStatus
        agent = MLStrategyAgent()

        context = MLStrategyContext(
            task_spec=state.task_spec,
            data_profile=state.data_profile,
            data_quality_report=state.data_quality_report,
            eda_findings=state.eda_findings,
            experiment_history=state.experiment_ledger,
        )

        strategy_plan = agent.plan(context)

        if strategy_plan.strategy_status == StrategyStatus.BLOCKED:
            patch = StatePatch(
                author_component="Orchestrator",
                target_phase=LifecyclePhase.TERMINATED,
                termination_reason=f"ML Strategy Blocked: {strategy_plan.problem_summary}",
                is_terminal=True,
            )
            return apply_state_patch(state, patch)

        # Convert StrategyPlan experiments to ExperimentConfigurations for execution queue
        configs = []
        for exp in strategy_plan.experiment_plan:
            configs.append(
                ExperimentConfiguration(
                    experiment_id=exp.experiment_id,
                    model_family=exp.model_family.value,
                    model_class_name=exp.model_class_name,
                    hyperparameters=exp.hyperparameters,
                    preprocessing_steps=exp.preprocessing_steps,
                    seed=exp.seed,
                    timeout_seconds=exp.timeout_seconds,
                )
            )

        patch = StatePatch(
            author_component="MLStrategist",
            new_experiments=configs,
            new_assumptions=strategy_plan.assumptions,
            target_phase=LifecyclePhase.DISPATCH,
        )
        return apply_state_patch(state, patch)

    def dispatch_and_execute_node(self, state: GlobalState) -> GlobalState:
        """Execute queued experiments using capability-isolated Model Specialists and ExperimentRunner."""
        if not state.experiment_queue:
            patch = StatePatch(
                author_component="Orchestrator",
                target_phase=LifecyclePhase.EVALUATE,
            )
            return apply_state_patch(state, patch)

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from amea.experiments.runner import ExperimentRunner
        from amea.model_specialists.registry import ModelSpecialistRegistry
        from amea.ml_strategy.models import ExperimentSpecification, ModelFamily

        registry = ModelSpecialistRegistry()
        runner = ExperimentRunner()
        records: List[RegisteredExperimentRecord] = []
        
        # Use cleaned dataset path if created by DataCleaningAgent
        cleaned_ds = state.data_quality_report.get("cleaned_dataset_path") if state.data_quality_report else None
        dataset_path = cleaned_ds if (cleaned_ds and Path(cleaned_ds).exists()) else (state.data_profile.dataset_path if state.data_profile else "data.csv")

        def run_single_experiment(exp_config) -> RegisteredExperimentRecord:
            # 1. Resolve Model Specialist via Registry
            specialist = registry.get_specialist(exp_config.model_family)
            if not specialist:
                specialist = registry.get_specialist("LinearModel")

            family_map = {
                "LinearModel": ModelFamily.LINEAR_MODEL,
                "RandomForest": ModelFamily.RANDOM_FOREST,
                "GradientBoosting": ModelFamily.GRADIENT_BOOSTING,
                "TabularNeuralNet": ModelFamily.TABULAR_NEURAL_NET,
            }
            family_enum = family_map.get(exp_config.model_family, ModelFamily.LINEAR_MODEL)

            # 2. Build ExperimentSpecification for specialist
            exp_spec = ExperimentSpecification(
                experiment_id=exp_config.experiment_id,
                hypothesis=f"Evaluate {exp_config.model_family} capability",
                model_family=family_enum,
                model_class_name=exp_config.model_class_name,
                preprocessing_steps=exp_config.preprocessing_steps,
                hyperparameters=exp_config.hyperparameters,
                seed=exp_config.seed,
                timeout_seconds=exp_config.timeout_seconds,
            )

            # 3. Model Specialist compiles ModelExecutionConfiguration
            exec_config = specialist.prepare_execution(
                exp_spec=exp_spec,
                task_spec=state.task_spec,
                dataset_path=dataset_path,
            )

            # 4. Experiment Runner executes in isolated workspace
            res = runner.run_experiment(exec_config)

            return RegisteredExperimentRecord(
                experiment_id=exp_config.experiment_id,
                model_family=exp_config.model_family,
                hyperparameters=exp_config.hyperparameters,
                cv_metrics_mean=res.cv_metrics_mean,
                cv_metrics_std=res.cv_metrics_std,
                train_metrics_mean=res.train_metrics_mean,
                training_duration_sec=res.resource_usage.duration_seconds,
                inference_latency_ms=res.inference_latency_ms,
                peak_memory_mb=res.resource_usage.peak_memory_mb,
                exit_code=res.exit_code,
                error_message=res.error.message if res.error else None,
            )

        # Execute candidate models concurrently within compute budget
        max_workers = min(4, max(1, len(state.experiment_queue)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(run_single_experiment, exp_cfg) for exp_cfg in state.experiment_queue]
            for fut in as_completed(futures):
                records.append(fut.result())

        # Sort records by experiment_id for deterministic ordering
        records.sort(key=lambda r: r.experiment_id)

        patch = StatePatch(
            author_component="ExperimentRunner",
            new_experiment_records=records,
            target_phase=LifecyclePhase.EVALUATE,
        )
        return apply_state_patch(state, patch)

    def evaluate_node(self, state: GlobalState) -> GlobalState:
        """Audit completed experiments via EvaluationAgent and select winner via JudgeAgent."""
        if not state.experiment_ledger:
            patch = StatePatch(
                author_component="Orchestrator",
                target_phase=LifecyclePhase.TERMINATED,
                termination_reason="No completed experiments to evaluate",
                is_terminal=True,
            )
            return apply_state_patch(state, patch)

        from amea.evaluation.auditor import EvaluationAgent
        from amea.evaluation.judge import JudgeAgent

        # 1. Evaluation Agent audits completed experiments for leakage and overfitting
        audits = EvaluationAgent.audit(
            records=state.experiment_ledger,
            primary_metric_name=state.task_spec.primary_metric,
            optimization_direction=state.task_spec.optimization_direction,
        )

        # Convert audit reports to state-compatible format
        state_audits: Dict[str, EvaluationAuditReport] = {}
        for exp_id, a in audits.items():
            state_audits[exp_id] = EvaluationAuditReport(
                experiment_id=exp_id,
                is_leakage_suspected=a.is_leakage_suspected,
                overfitting_gap=a.overfitting_gap,
                is_stable=a.is_stable,
                beats_baseline=a.beats_baseline,
                audit_passed=a.audit_passed,
            )

        # 2. Judge Agent performs Pareto analysis and selects optimal model
        best_candidate, decision = JudgeAgent.evaluate(
            records=state.experiment_ledger,
            audits=audits,
            primary_metric_name=state.task_spec.primary_metric,
            optimization_direction=state.task_spec.optimization_direction,
        )

        patch = StatePatch(
            author_component="JudgeAgent",
            new_audit_reports=state_audits,
            best_candidate=best_candidate,
            judge_decision=decision,
            target_phase=LifecyclePhase.BUILD,
        )
        return apply_state_patch(state, patch)

    def build_node(self, state: GlobalState) -> GlobalState:
        """Synthesize modular production Python code files via CodeSynthesisAgent."""
        if not state.best_candidate:
            patch = StatePatch(
                author_component="Orchestrator",
                target_phase=LifecyclePhase.TERMINATED,
                termination_reason="No selected best model candidate available for code synthesis.",
                is_terminal=True,
            )
            return apply_state_patch(state, patch)

        from amea.code_synthesis.agent import CodeSynthesisAgent
        from amea.code_synthesis.models import CodeSynthesisContext

        agent = CodeSynthesisAgent()
        context = CodeSynthesisContext(
            task_spec=state.task_spec,
            data_profile=state.data_profile,
            eda_report=None,
            cleaned_data_artifact=None,
            strategy_plan=None,
            best_candidate=state.best_candidate,
            judge_decision=state.judge_decision,
        )

        generated = agent.synthesize(context)

        artifacts = GeneratedCodeArtifacts(
            files=generated.files,
            entrypoint="train.py",
            target_environment="python",
        )

        patch = StatePatch(
            author_component="CodeSynthesisAgent",
            code_artifacts=artifacts,
            target_phase=LifecyclePhase.VERIFY,
        )
        return apply_state_patch(state, patch)

    def verify_node(self, state: GlobalState) -> GlobalState:
        """Execute synthesized code in sandbox to verify clean execution."""
        if not state.code_artifacts:
            patch = StatePatch(
                author_component="Orchestrator",
                target_phase=LifecyclePhase.TERMINATED,
                termination_reason="No code artifacts available to verify",
                is_terminal=True,
            )
            return apply_state_patch(state, patch)

        # Copy data file into sandbox execution context
        additional_files = dict(state.code_artifacts.files)
        if state.data_profile and Path(state.data_profile.dataset_path).exists():
            dataset_content = Path(state.data_profile.dataset_path).read_text(encoding="utf-8")
            additional_files[Path(state.data_profile.dataset_path).name] = dataset_content

        res = self.executor.execute_script(
            run_id="verify_synthesized_pipeline",
            script_content=state.code_artifacts.files["train.py"],
            additional_files=additional_files,
            timeout_seconds=120,
        )

        pipeline_res = PipelineExecutionResult(
            exit_code=res.exit_code,
            stdout=res.stdout,
            stderr=res.stderr,
            execution_duration_sec=res.duration_seconds,
            verified_metrics=res.metrics_extracted,
            artifacts_created=res.artifacts_created,
            is_verified=res.is_success,
        )

        patch = StatePatch(
            author_component="CodeExecutor",
            execution_result=pipeline_res,
            target_phase=LifecyclePhase.FINALIZE if res.is_success else LifecyclePhase.TERMINATED,
            termination_reason=None if res.is_success else f"Code verification failed with exit code {res.exit_code}: {res.stderr[:200]}",
            is_terminal=not res.is_success,
        )
        return apply_state_patch(state, patch)

    def finalize_node(self, state: GlobalState) -> GlobalState:
        """Synthesize final verified report and complete run."""
        best = state.best_candidate
        report = FinalReport(
            project_id=state.project_id,
            task_id=state.task_id,
            summary=f"Successfully built and verified ML pipeline using {best.model_family if best else 'Baseline'}.",
            best_model_family=best.model_family if best else "Unknown",
            best_metrics=best.cv_metrics_mean if best else {},
            total_experiments_run=len(state.experiment_ledger),
            total_duration_sec=sum(r.training_duration_sec for r in state.experiment_ledger),
            verified_code_paths=list(state.code_artifacts.files.keys()) if state.code_artifacts else [],
            remaining_risks=[],
        )

        patch = StatePatch(
            author_component="Finalizer",
            final_report=report,
            target_phase=LifecyclePhase.TERMINATED,
            termination_reason="Completed successfully with verified pipeline",
            is_terminal=True,
        )
        return apply_state_patch(state, patch)


