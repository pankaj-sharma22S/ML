"""Distribution distortion auditor evaluating pre vs post cleaning statistical shifts."""

from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from amea.data_validation.models import (
    CheckStatus,
    DistributionShiftMetric,
    ValidationCheckResult,
)


class DistributionAuditor:
    """Audits whether data cleaning operations introduced unexpected distribution distortions."""

    @staticmethod
    def audit(
        raw_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        ks_distortion_pvalue_threshold: float = 1e-4,
    ) -> Tuple[List[DistributionShiftMetric], List[ValidationCheckResult]]:
        shift_metrics: List[DistributionShiftMetric] = []
        check_results: List[ValidationCheckResult] = []

        common_numeric = [
            c for c in cleaned_df.columns
            if c in raw_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[c]) and pd.api.types.is_numeric_dtype(raw_df[c])
        ]

        severely_distorted_cols: List[str] = []

        for col in common_numeric:
            raw_vals = raw_df[col].dropna().values
            clean_vals = cleaned_df[col].dropna().values

            if len(raw_vals) < 5 or len(clean_vals) < 5:
                continue

            raw_mean = float(np.mean(raw_vals))
            clean_mean = float(np.mean(clean_vals))
            raw_med = float(np.median(raw_vals))
            clean_med = float(np.median(clean_vals))
            raw_std = float(np.std(raw_vals))
            clean_std = float(np.std(clean_vals))

            # Mean percentage difference
            mean_diff_pct = float(abs(clean_mean - raw_mean) / max(1e-6, abs(raw_mean))) if abs(raw_mean) > 1e-6 else 0.0

            # Kolmogorov-Smirnov 2-sample test
            ks_res = stats.ks_2samp(raw_vals, clean_vals)
            ks_stat = float(ks_res.statistic)
            ks_pval = float(ks_res.pvalue)

            # Severe distortion flagged if KS stat is very high (> 0.35) and p-value is extremely small (< 1e-4)
            is_distorted = (ks_stat > 0.35 and ks_pval < ks_distortion_pvalue_threshold)
            if is_distorted:
                severely_distorted_cols.append(col)

            shift_metrics.append(
                DistributionShiftMetric(
                    column_name=col,
                    raw_mean=round(raw_mean, 4),
                    clean_mean=round(clean_mean, 4),
                    mean_diff_pct=round(mean_diff_pct, 4),
                    raw_median=round(raw_med, 4),
                    clean_median=round(clean_med, 4),
                    raw_std=round(raw_std, 4),
                    clean_std=round(clean_std, 4),
                    ks_statistic=round(ks_stat, 4),
                    ks_pvalue=round(ks_pval, 6),
                    is_severely_distorted=is_distorted,
                )
            )

        # Generate check result
        if severely_distorted_cols:
            check_results.append(
                ValidationCheckResult(
                    check_name="distribution_preservation_audit",
                    category="distribution",
                    status=CheckStatus.WARN,
                    message=f"Distribution shifts detected in columns: {severely_distorted_cols}. Verify imputation/clipping parameters.",
                    evidence={"distorted_columns": severely_distorted_cols},
                    is_blocking=False,
                )
            )
        else:
            check_results.append(
                ValidationCheckResult(
                    check_name="distribution_preservation_audit",
                    category="distribution",
                    status=CheckStatus.PASS,
                    message="All continuous feature distributions preserved within acceptable statistical bounds.",
                    evidence={"audited_columns_count": len(common_numeric)},
                    is_blocking=False,
                )
            )

        return shift_metrics, check_results
