"""Task Arbitrator resolving true ML problem type by reconciling intent with data evidence."""

from typing import List, Optional, Tuple
import pandas as pd

from amea.core.state import TaskType
from amea.problem_understanding.models import ConflictFinding, IntentAnalysis, IntentCategory


class TaskArbitrator:
    """Arbitrates the actual mathematical TaskType using empirical data distributions."""

    @staticmethod
    def arbitrate(
        intent: IntentAnalysis,
        df: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
    ) -> Tuple[TaskType, str, List[ConflictFinding], List[str]]:
        conflicts: List[ConflictFinding] = []
        assumptions: List[str] = []

        target = target_column or intent.mentioned_target_candidate

        # If no dataframe available, fallback to pure intent
        if df is None or len(df) == 0:
            if intent.primary_intent == IntentCategory.REGRESSION:
                return TaskType.REGRESSION, target or "target", conflicts, ["Assumed regression task based purely on prompt keywords"]
            elif intent.primary_intent == IntentCategory.FORECASTING:
                return TaskType.TIME_SERIES, target or "target", conflicts, ["Assumed time-series task based purely on prompt keywords"]
            return TaskType.BINARY_CLASSIFICATION, target or "target", conflicts, ["Assumed binary classification default"]

        # 1. Resolve Target Column
        if not target or target not in df.columns:
            # Check candidate match
            matched = [c for c in df.columns if target and target.lower() in c.lower()]
            if matched:
                resolved_target = matched[0]
                assumptions.append(f"Fuzzy-matched target candidate '{target}' to dataset column '{resolved_target}'.")
                target = resolved_target
            else:
                # Default: last column
                target = df.columns[-1]
                assumptions.append(f"No explicit target specified; selected last column '{target}' as target candidate.")

        # 2. Inspect Target Series Empirical Properties
        target_series = df[target].dropna()
        distinct_count = int(target_series.nunique())
        is_numeric = pd.api.types.is_numeric_dtype(target_series)

        # 3. Determine TaskType
        if distinct_count == 2:
            resolved_task = TaskType.BINARY_CLASSIFICATION
            if intent.primary_intent == IntentCategory.REGRESSION:
                conflicts.append(
                    ConflictFinding(
                        conflict_type="task_type_mismatch",
                        severity="WARNING",
                        description=f"User requested regression, but target '{target}' contains exactly 2 distinct values ({list(target_series.unique())}). Arbitrated to BINARY_CLASSIFICATION.",
                        evidence={"distinct_count": 2, "unique_values": list(map(str, target_series.unique()[:2]))},
                        resolution="Arbitrated to BINARY_CLASSIFICATION for mathematical validity.",
                    )
                )

        elif 2 < distinct_count <= 20 and (not is_numeric or pd.api.types.is_integer_dtype(target_series) or target_series.dtype == "object"):
            resolved_task = TaskType.MULTICLASS_CLASSIFICATION
            if intent.primary_intent == IntentCategory.REGRESSION and distinct_count < 10:
                conflicts.append(
                    ConflictFinding(
                        conflict_type="task_type_mismatch",
                        severity="WARNING",
                        description=f"Target '{target}' has low discrete cardinality ({distinct_count} distinct categories). Arbitrated to MULTICLASS_CLASSIFICATION.",
                        evidence={"distinct_count": distinct_count},
                        resolution="Arbitrated to MULTICLASS_CLASSIFICATION.",
                    )
                )

        elif is_numeric and distinct_count > 20:
            resolved_task = TaskType.REGRESSION
            if intent.primary_intent == IntentCategory.CLASSIFICATION:
                conflicts.append(
                    ConflictFinding(
                        conflict_type="task_type_mismatch",
                        severity="WARNING",
                        description=f"User requested classification, but target '{target}' has {distinct_count} continuous numeric values. Arbitrated to REGRESSION.",
                        evidence={"distinct_count": distinct_count, "dtype": str(target_series.dtype)},
                        resolution="Arbitrated to REGRESSION.",
                    )
                )

        else:
            resolved_task = TaskType.BINARY_CLASSIFICATION

        # Check temporal intent vs data presence
        if intent.primary_intent == IntentCategory.FORECASTING:
            date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in c.lower() or "time" in c.lower()]
            if date_cols:
                resolved_task = TaskType.TIME_SERIES
            else:
                conflicts.append(
                    ConflictFinding(
                        conflict_type="temporal_mismatch",
                        severity="WARNING",
                        description="User requested forecasting/time-series, but no datetime columns were detected in dataset.",
                        evidence={"columns": list(df.columns)},
                        resolution="Retained standard tabular formulation.",
                    )
                )

        return resolved_task, target, conflicts, assumptions
