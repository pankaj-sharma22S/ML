import argparse
import sys
from pathlib import Path

# Add src to sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amea.core.config import ProjectConfig, ComputeBudget
from amea.orchestrator.runner import OrchestratorRunner


def main():
    parser = argparse.ArgumentParser(description="Autonomous ML Engineer Agent (AMEA)")
    parser.add_argument("--request", type=str, default="Train a baseline classifier for target prediction", help="ML Engineering objective")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default=None, help="Target column name")
    parser.add_argument("--max-experiments", type=int, default=5, help="Experiment compute budget")
    args = parser.parse_args()

    config = ProjectConfig(
        project_id="cli-run",
        budget=ComputeBudget(max_experiments=args.max_experiments),
    )

    runner = OrchestratorRunner(config=config)
    print(f"[*] Starting Autonomous ML Engineer Orchestrator...")
    print(f"[*] User Request: {args.request}")
    if args.data:
        print(f"[*] Dataset: {args.data}")

    state = runner.run_task(
        user_request=args.request,
        dataset_path=args.data,
        target_column=args.target,
    )

    print("\n" + "=" * 60)
    print(f"[*] Execution Completed: Terminal Phase = {state.current_phase.value}")
    if state.termination_reason:
        print(f"[*] Status Message: {state.termination_reason}")

    if state.best_candidate:
        print(f"[*] Best Candidate Model: {state.best_candidate.model_family}")
        print(f"[*] Best Metrics: {state.best_candidate.cv_metrics_mean}")

    if state.code_artifacts:
        print(f"[*] Generated Code Files: {list(state.code_artifacts.files.keys())}")

    if state.final_report:
        print(f"[*] Summary: {state.final_report.summary}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
