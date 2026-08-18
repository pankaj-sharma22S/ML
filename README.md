# Autonomous ML Engineer Agent (AMEA)

An enterprise-grade, capability-based Autonomous Machine Learning Engineering system designed with strict multi-agent separation of concerns.

---

## 📁 Repository Structure

```text
.
├── src/
│   └── amea/                           # Pure production source and agent implementations
│       ├── core/                       # State schemas, reducers, capability registry, events
│       ├── data_intelligence/          # Profiling, quality auditing, leakage prevention
│       ├── eda/                        # Distribution, outlier, categorical & relationship analysis
│       ├── data_cleaning/              # Reproducible, scikit-learn compatible cleaning pipeline
│       ├── data_validation/            # Schema, integrity, distribution & quality gate audits
│       ├── problem_understanding/      # Intent parsing, task arbitration, metric selection
│       ├── ml_strategy/                # Model selection, baseline design, feature hypotheses, budgeting
│       ├── model_specialists/          # Linear, Tree, Boosting, and Neural Specialist Agents
│       ├── experiments/                # Isolated sandbox ExperimentRunner and schemas
│       ├── evaluation/                 # Overfit/leakage auditor and Pareto Judge Agent
│       ├── orchestrator/               # Central DAG Orchestrator and state machine nodes
│       ├── execution/                  # Workspace isolation, security boundary, subprocess runner
│       └── persistence/                # State checkpointer and write-ahead log (WAL)
├── tests/                              # Comprehensive test suite (62+ unit and integration tests)
├── data/                               # Versioned benchmark and demo datasets
│   ├── README.md
│   └── sample_churn.csv                # Sample classification demo dataset
├── pyproject.toml                      # Package build configuration and dependencies
├── .gitignore                          # Comprehensive exclusion for caches, runtime dirs & datasets
└── README.md                           # Project documentation
```

### 🔒 Runtime Artifacts & Workspaces (Ignored by Git)
When AMEA runs, all intermediate artifacts, state checkpoints, logs, and sandbox workspaces are stored in dedicated runtime directories outside `src/` that are excluded from version control:
- `.amea_project/`: Project checkpoints, state write-ahead logs, and serializations.
- `.amea_project/experiments/`: Isolated per-experiment sandboxes, logs, and trained model artifacts.
- `.amea_sandboxes/`: Ephemeral code verification sandboxes.
- `workspace/` / `artifacts/`: Generated user-facing reports and verified standalone pipelines.

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
git clone <repository-url>
cd ML
pip install -e .
```

### 2. Running Automated Tests
```bash
pytest tests -v
```

### 3. Running an Autonomous ML Task
```bash
python src/amea/main.py --request "Train churn classifier with model specialists" --data "data/sample_churn.csv" --target "churn"
```
