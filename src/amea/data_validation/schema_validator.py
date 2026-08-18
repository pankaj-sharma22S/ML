"""Schema and column consistency validator."""

from typing import List, Optional, Set
import pandas as pd

from amea.data_validation.models import CheckStatus, ValidationCheckResult


class SchemaValidator:
    """Audits column presence, types, and target variable integrity."""

    @staticmethod
    def validate(
        raw_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        target_column: Optional[str] = None,
        expected_dropped_columns: Optional[Set[str]] = None,
    ) -> List[ValidationCheckResult]:
        results: List[ValidationCheckResult] = []
        dropped_expected = expected_dropped_columns or set()

        # 1. Target Column Existence Check
        if target_column:
            if target_column not in cleaned_df.columns:
                results.append(
                    ValidationCheckResult(
                        check_name="target_column_existence",
                        category="schema",
                        status=CheckStatus.FAIL,
                        message=f"CRITICAL: Target column '{target_column}' is missing from the cleaned dataset.",
                        evidence={"target_column": target_column, "cleaned_columns": list(cleaned_df.columns)},
                        is_blocking=True,
                    )
                )
            else:
                results.append(
                    ValidationCheckResult(
                        check_name="target_column_existence",
                        category="schema",
                        status=CheckStatus.PASS,
                        message=f"Target column '{target_column}' verified present.",
                        evidence={"target_column": target_column},
                        is_blocking=False,
                    )
                )

        # 2. Unexpected Missing Columns Check
        raw_cols = set(raw_df.columns)
        clean_cols = set(cleaned_df.columns)
        missing_cols = raw_cols - clean_cols - dropped_expected

        if missing_cols:
            results.append(
                ValidationCheckResult(
                    check_name="unexpected_column_loss",
                    category="schema",
                    status=CheckStatus.FAIL,
                    message=f"Columns were dropped without explicit approval: {list(missing_cols)}",
                    evidence={"missing_columns": list(missing_cols)},
                    is_blocking=True,
                )
            )
        else:
            results.append(
                ValidationCheckResult(
                    check_name="unexpected_column_loss",
                    category="schema",
                    status=CheckStatus.PASS,
                    message="All retained columns match expected approved feature sets.",
                    evidence={"retained_columns_count": len(clean_cols)},
                    is_blocking=False,
                )
            )

        # 3. Non-Empty Features Check
        feature_cols = [c for c in cleaned_df.columns if c != target_column]
        if len(feature_cols) == 0:
            results.append(
                ValidationCheckResult(
                    check_name="feature_dimension_check",
                    category="schema",
                    status=CheckStatus.FAIL,
                    message="CRITICAL: Cleaned dataset contains 0 feature columns.",
                    evidence={"feature_count": 0},
                    is_blocking=True,
                )
            )
        else:
            results.append(
                ValidationCheckResult(
                    check_name="feature_dimension_check",
                    category="schema",
                    status=CheckStatus.PASS,
                    message=f"Cleaned dataset contains {len(feature_cols)} feature columns.",
                    evidence={"feature_count": len(feature_cols)},
                    is_blocking=False,
                )
            )

        return results
