"""Tree and Random Forest Model Specialist Agent."""

import json
from pathlib import Path
from typing import Any, Dict

from amea.core.state import MLTaskSpecification, TaskType
from amea.experiments.models import ModelExecutionConfiguration
from amea.ml_strategy.models import ExperimentSpecification, ModelFamily
from amea.model_specialists.base import ModelCapability, ModelSpecialistBase, SpecialistValidationResult


class TreeModelSpecialistAgent(ModelSpecialistBase):
    """Specialist managing decision tree and bootstrap aggregated random forest models."""

    def capabilities(self) -> ModelCapability:
        return ModelCapability(
            model_family=ModelFamily.RANDOM_FOREST,
            supported_task_types=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION, TaskType.REGRESSION],
            strengths=["Non-linear decision surfaces", "Feature interaction capture", "Robust against monotonic scale changes", "Resistant to outliers"],
            limitations=["Large memory footprint for deep ensembles", "Cannot extrapolate beyond training domain"],
            requires_scaling=False,
            handles_missing_values_natively=False,
            gpu_accelerated=False,
        )

    def validate_experiment(
        self,
        exp_spec: ExperimentSpecification,
        task_spec: MLTaskSpecification,
    ) -> SpecialistValidationResult:
        caps = self.capabilities()
        if task_spec.task_type not in caps.supported_task_types:
            return SpecialistValidationResult(
                is_compatible=False,
                status="INCOMPATIBLE",
                rejection_reason=f"TreeModelSpecialist does not support task type '{task_spec.task_type.value}'.",
                required_action="Route experiment to a specialist supporting this task type.",
            )
        return SpecialistValidationResult(is_compatible=True, status="COMPATIBLE")

    def prepare_execution(
        self,
        exp_spec: ExperimentSpecification,
        task_spec: MLTaskSpecification,
        dataset_path: str,
    ) -> ModelExecutionConfiguration:
        data_path_clean = Path(dataset_path).as_posix()
        target_col = task_spec.target_column
        primary_metric = task_spec.primary_metric
        is_classification = (task_spec.task_type != TaskType.REGRESSION)
        model_cls = "RandomForestClassifier" if is_classification else "RandomForestRegressor"

        script = f'''import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
from sklearn.impute import SimpleImputer

# 1. Load data
df = pd.read_csv(r"{data_path_clean}")
target = "{target_col}"
X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
y = df[target]

# 2. Impute (Trees do not require scaling)
imputer = SimpleImputer(strategy='median')
X_clean = imputer.fit_transform(X)

# 3. Model setup
model = {model_cls}(**{json.dumps(exp_spec.hyperparameters)})

# 4. Cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state={exp_spec.seed}) if '{is_classification}' == 'True' else KFold(n_splits=3, shuffle=True, random_state={exp_spec.seed})
scores = []

for train_idx, val_idx in cv.split(X_clean, y):
    X_tr, X_val = X_clean[train_idx], X_clean[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_tr, y_tr)
    if '{is_classification}' == 'True':
        preds = model.predict(X_val)
        scores.append(float(accuracy_score(y_val, preds)))
    else:
        preds = model.predict(X_val)
        scores.append(float(mean_squared_error(y_val, preds)))

metric_val = float(np.mean(scores))
metrics = {{"{primary_metric}": metric_val}}
print(f"__AMEA_METRICS__={{json.dumps(metrics)}}")
'''

        return ModelExecutionConfiguration(
            experiment_id=exp_spec.experiment_id,
            model_family=exp_spec.model_family.value,
            model_class_name=model_cls,
            script_content=script,
            hyperparameters=exp_spec.hyperparameters,
            preprocessing_steps=["SimpleImputer"],
            dataset_path=dataset_path,
            target_column=target_col,
            primary_metric=primary_metric,
            secondary_metrics=task_spec.secondary_metrics,
            task_type=task_spec.task_type.value,
            seed=exp_spec.seed,
            timeout_seconds=exp_spec.timeout_seconds,
        )
