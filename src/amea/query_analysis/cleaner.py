"""Evidence-based targeted data cleaning engine."""

from typing import Dict, List, Tuple
import pandas as pd
from amea.query_analysis.schemas import CleaningAction, DataQualityIssue, DatasetProfile, QueryIntent
from amea.query_analysis.ingestion import IngestionFileRecord


class EvidenceBasedCleaner:
    """Detects quality issues and executes targeted, query-justified cleaning operations."""

    @classmethod
    def audit_quality(
        cls,
        record: IngestionFileRecord,
        profile: DatasetProfile,
        intent: QueryIntent,
    ) -> List[DataQualityIssue]:
        """Audit dataset for quality issues relevant to the user query."""
        df = record.df
        issues: List[DataQualityIssue] = []

        # 1. Duplicate rows
        if profile.duplicate_rows_count > 0:
            # Duplicates impact aggregation & rankings
            impacts = intent.primary_intent in ["aggregation", "ranking", "trend_analysis"] or \
                      any(i in ["aggregation", "ranking"] for i in intent.secondary_intents)
            issues.append(DataQualityIssue(
                dataset_id=record.dataset_id,
                issue_type="duplicate_rows",
                affected_columns=list(df.columns),
                affected_rows_count=profile.duplicate_rows_count,
                severity="IMPORTANT" if impacts else "LOW",
                impacts_user_query=impacts,
                description=f"Found {profile.duplicate_rows_count} duplicate rows that could inflate aggregations.",
            ))

        # 2. Missing values in target metrics
        for col, missing_cnt in profile.missing_summary.items():
            if missing_cnt > 0:
                is_metric = any(m.lower() in col.lower() for m in intent.target_metrics)
                issues.append(DataQualityIssue(
                    dataset_id=record.dataset_id,
                    issue_type="missing_values",
                    affected_columns=[col],
                    affected_rows_count=missing_cnt,
                    severity="CRITICAL" if is_metric else "LOW",
                    impacts_user_query=is_metric,
                    description=f"Column '{col}' has {missing_cnt} missing values ({missing_cnt / len(df):.1%}).",
                ))

        return issues

    @classmethod
    def clean_dataset(
        cls,
        record: IngestionFileRecord,
        issues: List[DataQualityIssue],
    ) -> Tuple[pd.DataFrame, List[CleaningAction]]:
        """Applies justified cleaning actions on a copy of DataFrame."""
        cleaned_df = record.df.copy()
        actions: List[CleaningAction] = []

        for issue in issues:
            if not issue.impacts_user_query:
                continue

            if issue.issue_type == "duplicate_rows":
                before_rows = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                after_rows = len(cleaned_df)

                actions.append(CleaningAction(
                    dataset_id=record.dataset_id,
                    issue="duplicate_rows",
                    affected_columns=issue.affected_columns,
                    affected_rows=issue.affected_rows_count,
                    operation="drop_duplicates",
                    reason="Removed duplicate rows to prevent distorted aggregations and inflated metric sums.",
                    before_stats={"row_count": before_rows},
                    after_stats={"row_count": after_rows},
                ))

            elif issue.issue_type == "missing_values":
                for col in issue.affected_columns:
                    if col in cleaned_df.columns:
                        before_nulls = int(cleaned_df[col].isnull().sum())
                        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                            median_val = float(cleaned_df[col].median())
                            cleaned_df[col] = cleaned_df[col].fillna(median_val)
                            after_nulls = int(cleaned_df[col].isnull().sum())

                            actions.append(CleaningAction(
                                dataset_id=record.dataset_id,
                                issue="missing_values",
                                affected_columns=[col],
                                affected_rows=issue.affected_rows_count,
                                operation="impute_median",
                                reason=f"Imputed {before_nulls} missing values in '{col}' using median ({median_val:.2f}).",
                                before_stats={"null_count": before_nulls},
                                after_stats={"null_count": after_nulls},
                            ))

        return cleaned_df, actions
