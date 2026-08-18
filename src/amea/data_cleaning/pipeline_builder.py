"""Pipeline Builder constructing reproducible CleaningPipelines from CleaningPlans."""

from typing import List
import pandas as pd
from sklearn.pipeline import Pipeline

from amea.data_cleaning.models import CleaningActionType, CleaningPlan
from amea.data_cleaning.transformers import (
    ColumnDropperTransformer,
    OutlierClipperTransformer,
    RareCategoryGrouperTransformer,
    AdaptiveImputerTransformer,
)
from amea.data_intelligence.models import DataTreatmentCandidate


class CleaningPipelineBuilder:
    """Builds a scikit-learn compatible Pipeline from approved candidate treatments or CleaningPlan."""

    @staticmethod
    def build_from_treatments(candidates: List[DataTreatmentCandidate]) -> Pipeline:
        steps = []

        # 1. Dropping Stage
        drop_cols = []
        for c in candidates:
            if c.treatment_type == "drop_feature":
                drop_cols.extend(c.target_columns)
        if drop_cols:
            steps.append(("dropper", ColumnDropperTransformer(columns_to_drop=list(set(drop_cols)))))

        # 2. Rare Category Grouping Stage
        cat_cols = []
        for c in candidates:
            if "rare" in c.strategy_id or "cardinality" in c.strategy_id:
                cat_cols.extend(c.target_columns)
        if cat_cols:
            steps.append(("rare_grouper", RareCategoryGrouperTransformer(columns=list(set(cat_cols)))))

        # 3. Outlier Clipping Stage
        outlier_cols = []
        for c in candidates:
            if c.treatment_type in ("outlier_handling", "scaling") and "robust" in c.strategy_id:
                outlier_cols.extend(c.target_columns)
        if outlier_cols:
            steps.append(("clipper", OutlierClipperTransformer(columns=list(set(outlier_cols)))))

        # 4. Adaptive Imputation Stage
        steps.append(("imputer", AdaptiveImputerTransformer()))

        return Pipeline(steps)

    @staticmethod
    def build_from_plan(plan: CleaningPlan) -> Pipeline:
        steps = []
        for i, action in enumerate(plan.actions):
            step_name = f"step_{i}_{action.action_type.value.lower()}"
            if action.action_type == CleaningActionType.DROP_COLUMNS:
                steps.append((step_name, ColumnDropperTransformer(columns_to_drop=action.target_columns)))
            elif action.action_type == CleaningActionType.CLIP_OUTLIERS:
                low = action.parameters.get("lower_quantile", 0.01)
                high = action.parameters.get("upper_quantile", 0.99)
                steps.append((step_name, OutlierClipperTransformer(lower_quantile=low, upper_quantile=high, columns=action.target_columns)))
            elif action.action_type == CleaningActionType.GROUP_RARE_CATEGORIES:
                min_freq = action.parameters.get("min_frequency", 0.01)
                steps.append((step_name, RareCategoryGrouperTransformer(min_frequency=min_freq, columns=action.target_columns)))
            elif action.action_type in (CleaningActionType.IMPUTE_MEDIAN, CleaningActionType.IMPUTE_MODE):
                steps.append((step_name, AdaptiveImputerTransformer()))

        if not steps:
            steps.append(("passthrough_imputer", AdaptiveImputerTransformer()))

        return Pipeline(steps)
