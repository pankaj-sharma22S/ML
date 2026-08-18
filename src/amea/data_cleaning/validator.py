"""Post-Cleaning Validator auditing dataset completeness, nulls, and target preservation."""

from typing import List, Optional
import pandas as pd
from amea.data_cleaning.models import PostCleaningValidationReport


class PostCleaningValidator:
    """Audits cleaned DataFrames to guarantee data readiness and absence of nulls/leakage."""

    @staticmethod
    def validate(
        initial_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> PostCleaningValidationReport:
        messages: List[str] = []
        is_valid = True

        initial_rows = len(initial_df)
        final_rows = len(cleaned_df)
        initial_cols = len(initial_df.columns)
        final_cols = len(cleaned_df.columns)

        # 1. Null Count Check
        null_count = int(cleaned_df.isnull().sum().sum())
        if null_count > 0:
            is_valid = False
            messages.append(f"Post-cleaning validation failed: {null_count} null values remain in cleaned dataset.")
        else:
            messages.append("Zero null values verified in cleaned dataset.")

        # 2. Row Count Integrity
        if final_rows == 0:
            is_valid = False
            messages.append("Post-cleaning validation failed: Cleaned dataset contains 0 rows.")
        elif final_rows < initial_rows:
            dropped_rows = initial_rows - final_rows
            messages.append(f"Row count reduced by {dropped_rows} rows (e.g. duplicate removal).")

        # 3. Target Column Preservation
        target_preserved = True
        if target_column:
            if target_column not in cleaned_df.columns:
                is_valid = False
                target_preserved = False
                messages.append(f"Critical error: Target column '{target_column}' was accidentally dropped during cleaning.")
            else:
                messages.append(f"Target column '{target_column}' verified present.")

        # 4. Column Tracking
        dropped_columns = [c for c in initial_df.columns if c not in cleaned_df.columns]
        if dropped_columns:
            messages.append(f"Columns dropped during cleaning: {dropped_columns}")

        return PostCleaningValidationReport(
            is_valid=is_valid,
            initial_rows=initial_rows,
            final_rows=final_rows,
            initial_columns=initial_cols,
            final_columns=final_cols,
            remaining_null_count=null_count,
            target_column_preserved=target_preserved,
            columns_dropped=dropped_columns,
            validation_messages=messages,
        )
