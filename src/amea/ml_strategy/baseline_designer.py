"""Task-specific baseline model designer."""

from amea.core.state import TaskType
from amea.ml_strategy.models import ModelCandidate, ModelFamily


class BaselineDesigner:
    """Designs mathematically appropriate baselines for supervised learning tasks."""

    @staticmethod
    def design_baseline(task_type: TaskType, random_seed: int = 42) -> ModelCandidate:
        if task_type == TaskType.BINARY_CLASSIFICATION:
            return ModelCandidate(
                candidate_id="baseline_logistic",
                model_family=ModelFamily.LINEAR_MODEL,
                model_class_name="LogisticRegression",
                rationale="L2-regularized logistic regression with median imputation and standardization serving as the primary linear reference.",
                default_hyperparameters={"C": 1.0, "max_iter": 500, "random_state": random_seed},
                priority=1,
            )
        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            return ModelCandidate(
                candidate_id="baseline_multinomial_logistic",
                model_family=ModelFamily.LINEAR_MODEL,
                model_class_name="LogisticRegression",
                rationale="Multinomial logistic regression serving as the linear baseline.",
                default_hyperparameters={"C": 1.0, "multi_class": "multinomial", "max_iter": 500, "random_state": random_seed},
                priority=1,
            )
        else:
            return ModelCandidate(
                candidate_id="baseline_ridge",
                model_family=ModelFamily.LINEAR_MODEL,
                model_class_name="Ridge",
                rationale="L2-regularized linear ridge regression with standardization as the continuous baseline.",
                default_hyperparameters={"alpha": 1.0, "random_state": random_seed},
                priority=1,
            )
