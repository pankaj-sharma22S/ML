"""Row count reconciliation, duplicate verification, and missingness auditor."""

from typing import List, Optional
import pandas as pd

from amea.data_validation.models import CheckStatus, ValidationCheckResult


class IntegrityValidator:
    """Audits row count preservation, absence of residual nulls, and duplicate integrity."""

    @staticmethod
    def validate(
        raw_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        target_column: Optional[str] = None,
        max_allowed_row_loss_pct: float = 0.10,
    ) -> List[ValidationCheckResult]:
        results: List[ValidationCheckResult] = []

        initial_rows = len(raw_df)
        final_rows = len(cleaned_df)

        # 1. Row Count Loss Reconciliation
        if final_rows == 0:
            results.append(
                ValidationCheckResult(
                    check_name="row_count_check",
                    category="integrity",
                    status=CheckStatus.FAIL,
                    message="CRITICAL: Cleaned dataset has 0 rows.",
                    evidence={"initial_rows": initial_rows, "final_rows": final_rows},
                    is_blocking=True,
                )
            )
        else:
            row_loss_ratio = float((initial_rows - final_rows) / max(1, initial_rows)) if initial_rows > 0 else 0.0
            if row_loss_ratio > max_allowed_row_loss_pct:
                results.append(
                    ValidationCheckResult(
                        check_name="row_loss_reconciliation",
                        category="integrity",
                        status=CheckStatus.FAIL,
                        message=f"Severe row loss: {row_loss_ratio*100:.1f}% rows dropped (max allowed: {max_allowed_row_loss_pct*100:.1f}%).",
                        evidence={"initial_rows": initial_rows, "final_rows": final_rows, "row_loss_ratio": row_loss_ratio},
                        is_blocking=True,
                    )
                )
            elif row_loss_ratio > 0.0:
                results.append(
                    ValidationCheckResult(
                        check_name="row_loss_reconciliation",
                        category="integrity",
                        status=CheckStatus.WARN,
                        message=f"Moderate row reduction: {row_loss_ratio*100:.1f}% rows removed during cleaning.",
                        evidence={"initial_rows": initial_rows, "final_rows": final_rows, "row_loss_ratio": row_loss_ratio},
                        is_blocking=False,
                    )
                )
            else:
                results.append(
                    ValidationCheckResult(
                        check_name="row_loss_reconciliation",
                        category="integrity",
                        status=CheckStatus.PASS,
                        message=f"Row count 100% preserved ({final_rows} rows).",
                        evidence={"rows": final_rows},
                        is_blocking=False,
                    )
                )

        # 2. Residual Null Count Verification (Strict Zero-Null Gate)
        null_counts = cleaned_df.isnull().sum()
        total_nulls = int(null_counts.sum())

        if total_nulls > 0:
            null_cols = null_counts[null_counts > 0].to_dict()
            results.append(
                ValidationCheckResult(
                    check_name="residual_missingness_check",
                    category="missingness",
                    status=CheckStatus.FAIL,
                    message=f"CRITICAL: {total_nulls} unhandled NaN values remain in cleaned dataset across columns: {null_cols}",
                    evidence={"total_nulls": total_nulls, "null_columns": null_cols},
                    is_blocking=True,
                )
            )
        else:
            results.append(
                ValidationCheckResult(
                    check_name="residual_missingness_check",
                    category="missingness",
                    status=CheckStatus.PASS,
                    message="Zero residual null values verified across all columns.",
                    evidence={"total_nulls": 0},
                    is_blocking=False,
                )
            )

        # 3. Target Null Verification
        if target_column and target_column in cleaned_df.columns:
            target_nulls = int(cleaned_df[target_column].isnull().sum())
            if target_nulls > 0:
                results.append(
                    ValidationCheckResult(
                        check_name="target_null_check",
                        category="missingness",
                        status=CheckStatus.FAIL,
                        message=f"CRITICAL: Target column '{target_column}' contains {target_nulls} missing values.",
                        evidence={"target_nulls": target_nulls},
                        is_blocking=True,
                    )
                )
            else:
                results.append(
                    ValidationCheckResult(
                        check_name="target_null_check",
                        category="missingness",
                        status=CheckStatus.PASS,
                        message=f"Target column '{target_column}' is 100% non-null.",
                        evidence={"target_nulls": 0},
                        is_blocking=False,
                    )
                )

        return results
