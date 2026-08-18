"""Evidence-driven Model Family Selector and Exclusion Engine."""

from typing import List, Optional, Tuple
from amea.core.state import DataProfile, TaskType
from amea.ml_strategy.models import ExcludedApproach, ModelCandidate, ModelFamily


class ModelSelector:
    """Selects high-utility model candidates and provides explicit mathematical exclusion reasons."""

    @staticmethod
    def select_candidates(
        task_type: TaskType,
        data_profile: Optional[DataProfile] = None,
        random_seed: int = 42,
    ) -> Tuple[List[ModelCandidate], List[ExcludedApproach]]:
        candidates: List[ModelCandidate] = []
        exclusions: List[ExcludedApproach] = []

        total_rows = data_profile.total_rows if data_profile else 1000
        total_cols = data_profile.total_columns if data_profile else 10

        is_classification = (task_type != TaskType.REGRESSION and task_type != TaskType.TIME_SERIES)

        # 1. Linear Candidate (Fast, Convex, Transparent Baseline)
        if is_classification:
            linear_class = "LogisticRegression"
            linear_params = {"C": 1.0, "max_iter": 500, "random_state": random_seed}
        else:
            linear_class = "Ridge"
            linear_params = {"alpha": 1.0, "random_state": random_seed}

        candidates.append(
            ModelCandidate(
                candidate_id="linear_baseline",
                model_family=ModelFamily.LINEAR_MODEL,
                model_class_name=linear_class,
                rationale="Convex, fast optimization providing an interpretable parametric reference.",
                default_hyperparameters=linear_params,
                priority=1,
            )
        )

        # 2. Random Forest Candidate (Ensemble Robust to Outliers & Non-linearities)
        if total_rows >= 30:
            if is_classification:
                rf_class = "RandomForestClassifier"
                rf_params = {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "random_state": random_seed}
            else:
                rf_class = "RandomForestRegressor"
                rf_params = {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "random_state": random_seed}

            candidates.append(
                ModelCandidate(
                    candidate_id="random_forest",
                    model_family=ModelFamily.RANDOM_FOREST,
                    model_class_name=rf_class,
                    rationale="Robust bagging ensemble that handles feature interactions and non-linear boundaries with low variance.",
                    default_hyperparameters=rf_params,
                    priority=2,
                )
            )
        else:
            exclusions.append(
                ExcludedApproach(
                    family_or_technique="RandomForest",
                    reason_for_exclusion=f"Dataset too small ({total_rows} rows < 30) for reliable bootstrap tree ensemble.",
                )
            )

        # 3. Gradient Boosting Candidate
        if total_rows >= 100:
            if is_classification:
                gb_class = "HistGradientBoostingClassifier"
                gb_params = {"max_iter": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": random_seed}
            else:
                gb_class = "HistGradientBoostingRegressor"
                gb_params = {"max_iter": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": random_seed}

            candidates.append(
                ModelCandidate(
                    candidate_id="gradient_boosting",
                    model_family=ModelFamily.GRADIENT_BOOSTING,
                    model_class_name=gb_class,
                    rationale="State-of-the-art tabular gradient boosting optimizer with native histogram binning.",
                    default_hyperparameters=gb_params,
                    priority=3,
                )
            )
        else:
            exclusions.append(
                ExcludedApproach(
                    family_or_technique="GradientBoosting",
                    reason_for_exclusion=f"Dataset size ({total_rows} rows < 100) insufficient for sequential gradient boosting without severe overfitting.",
                )
            )

        # 4. Neural Network Exclusions
        if total_rows < 5000:
            exclusions.append(
                ExcludedApproach(
                    family_or_technique="TabularNeuralNet",
                    reason_for_exclusion=f"Dataset size ({total_rows} rows < 5,000) is sample-inefficient for deep tabular architectures vs tree ensembles.",
                )
            )

        return candidates, exclusions
