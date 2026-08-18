"""Data Quality Auditor evaluating missingness, duplicates, outliers, and anomalies."""

from typing import Dict, List
import numpy as np
import pandas as pd
from scipy import stats

from amea.data_intelligence.models import (
    MissingnessFinding,
    MissingnessMechanism,
    OutlierFinding,
    QualityAuditReport,
)


class QualityAuditor:
    """Audits data quality, missingness patterns, and candidate anomalies."""

    @staticmethod
    def audit(df: pd.DataFrame) -> QualityAuditReport:
        total_rows = len(df)
        dup_count = int(df.duplicated().sum())
        dup_ratio = round(float(dup_count / total_rows), 4) if total_rows > 0 else 0.0

        constant_cols: List[str] = []
        quasi_constant_cols: List[str] = []
        missing_findings: List[MissingnessFinding] = []
        outlier_findings: List[OutlierFinding] = []

        # 1. Missingness & Variance Audits
        null_indicators = {}
        for col in df.columns:
            series = df[col]
            null_count = int(series.isnull().sum())
            null_ratio = float(null_count / total_rows) if total_rows > 0 else 0.0

            if null_count > 0:
                null_indicators[col] = series.isnull().astype(int)

            # Check constants
            distinct = int(series.nunique(dropna=True))
            if distinct <= 1:
                constant_cols.append(col)
            elif total_rows > 0:
                top_freq = series.value_counts(normalize=True, dropna=True).iloc[0]
                if top_freq >= 0.99:
                    quasi_constant_cols.append(col)

        # 2. Missingness Mechanism Heuristic
        for col, indicator in null_indicators.items():
            missing_count = int(indicator.sum())
            missing_ratio = round(float(missing_count / total_rows), 4)

            # Check correlation with other missing columns
            correlated_cols = []
            for other_col, other_ind in null_indicators.items():
                if other_col != col:
                    corr = float(np.corrcoef(indicator.values, other_ind.values)[0, 1])
                    if not np.isnan(corr) and abs(corr) > 0.6:
                        correlated_cols.append(other_col)

            # Candidate mechanism
            if missing_ratio > 0.5:
                mechanism = MissingnessMechanism.MNAR
                rec = f"High missingness ({missing_ratio*100:.1f}%). Evaluate missingness indicator feature or dropping if information is redundant."
            elif correlated_cols:
                mechanism = MissingnessMechanism.MAR
                rec = f"Missingness correlates with {correlated_cols}. Consider multivariate iterative imputation or tree-native NaN handling."
            else:
                mechanism = MissingnessMechanism.MCAR
                rec = "Missingness appears isolated/random. Simple imputation (median/mode) or tree-native handling is suitable."

            missing_findings.append(
                MissingnessFinding(
                    column_name=col,
                    missing_count=missing_count,
                    missing_ratio=missing_ratio,
                    candidate_mechanism=mechanism,
                    correlated_missing_columns=correlated_cols,
                    recommendation=rec,
                )
            )

        # 3. Numeric Outlier Diagnosis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            vals = df[col].dropna().values
            if len(vals) < 10:
                continue

            q25, q75 = np.percentile(vals, [25, 75])
            iqr = q75 - q25
            if iqr > 1e-9:
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                iqr_outliers = int(np.sum((vals < lower_bound) | (vals > upper_bound)))
            else:
                iqr_outliers = 0

            # Z-Score
            z_scores = np.abs(stats.zscore(vals)) if np.std(vals) > 1e-9 else np.zeros_like(vals)
            z_outliers = int(np.sum(z_scores > 3.0))

            outlier_ratio = round(float(iqr_outliers / len(vals)), 4)
            skewness = float(stats.skew(vals)) if np.std(vals) > 1e-9 else 0.0
            kurt = float(stats.kurtosis(vals)) if np.std(vals) > 1e-9 else 0.0

            if iqr_outliers > 0 or z_outliers > 0:
                is_severe = (outlier_ratio >= 0.05 and abs(skewness) > 2.0)
                note = f"{iqr_outliers} IQR outliers ({outlier_ratio*100:.1f}%), skewness={skewness:.2f}. "
                note += "Severe skew: recommend robust scaling or tree models." if is_severe else "Moderate outliers: retain or use robust scaling."

                outlier_findings.append(
                    OutlierFinding(
                        column_name=col,
                        iqr_outlier_count=iqr_outliers,
                        zscore_outlier_count=z_outliers,
                        outlier_ratio=outlier_ratio,
                        skewness=round(skewness, 2),
                        kurtosis=round(kurt, 2),
                        is_severe=is_severe,
                        evidence_note=note,
                    )
                )

        # 4. Composite Quality Score
        quality_score = 1.0
        if dup_ratio > 0:
            quality_score -= min(0.2, dup_ratio)
        if constant_cols:
            quality_score -= min(0.2, len(constant_cols) * 0.05)
        if any(m.missing_ratio > 0.3 for m in missing_findings):
            quality_score -= 0.15

        is_clean = (quality_score >= 0.85 and dup_count == 0)

        return QualityAuditReport(
            is_clean=is_clean,
            duplicate_rows_count=dup_count,
            duplicate_rows_ratio=dup_ratio,
            constant_columns=constant_cols,
            quasi_constant_columns=quasi_constant_cols,
            missingness_findings=missing_findings,
            outlier_findings=outlier_findings,
            quality_score=round(max(0.0, quality_score), 2),
        )
