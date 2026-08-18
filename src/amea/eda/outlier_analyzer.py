"""Outlier Analyzer categorizing statistical anomalies without naive deletion."""

from typing import List
import numpy as np
import pandas as pd
from scipy import stats

from amea.eda.models import EDAFinding, EDASeverity, OutlierCategory


class OutlierAnalyzer:
    """Classifies candidate outliers into semantic categories with actionable candidate treatments."""

    @staticmethod
    def analyze(df: pd.DataFrame) -> List[EDAFinding]:
        findings: List[EDAFinding] = []
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue

            vals = series.values
            q25, q75 = np.percentile(vals, [25, 75])
            iqr = q75 - q25
            if iqr <= 1e-9:
                continue

            lower_iqr = q25 - 1.5 * iqr
            upper_iqr = q75 + 1.5 * iqr
            iqr_outliers = int(np.sum((vals < lower_iqr) | (vals > upper_iqr)))
            iqr_ratio = float(iqr_outliers / len(vals))

            # Extreme 3.0 * IQR fence
            lower_extreme = q25 - 3.0 * iqr
            upper_extreme = q75 + 3.0 * iqr
            extreme_outliers = int(np.sum((vals < lower_extreme) | (vals > upper_extreme)))

            # MAD (Median Absolute Deviation)
            med = np.median(vals)
            mad = float(np.median(np.abs(vals - med)))
            mad_outliers = int(np.sum(np.abs(vals - med) > 3.5 * (mad * 1.4826))) if mad > 1e-9 else 0

            # Skewness
            skew_val = float(stats.skew(vals)) if np.std(vals) > 1e-9 else 0.0

            if iqr_outliers == 0:
                continue

            # Semantic Classification
            if extreme_outliers > 0 and (np.min(vals) < 0 and "age" in col.lower() or "count" in col.lower() and np.min(vals) < 0):
                category = OutlierCategory.LIKELY_INVALID
                sev = EDASeverity.CRITICAL
                diag = f"Values violate physical/domain bounds (e.g. negative values in {col})."
                strategies = ["ClipToValidBounds", "ImputeInvalidEntries", "VerifySourceLogic"]
            elif extreme_outliers > 0 and abs(skew_val) > 3.0:
                category = OutlierCategory.LEGITIMATE_EXTREME
                sev = EDASeverity.IMPORTANT
                diag = f"{extreme_outliers} extreme values appear to be legitimate natural heavy tails (skew={skew_val:.2f})."
                strategies = ["RobustScaler", "QuantileTransform", "KeepAndUseTreeModels", "WinsorizePercentile99"]
            elif iqr_ratio > 0.05:
                category = OutlierCategory.POTENTIALLY_INVALID
                sev = EDASeverity.IMPORTANT
                diag = f"High density of moderate outliers ({iqr_outliers} rows, {iqr_ratio*100:.1f}%)."
                strategies = ["Winsorization", "RobustScaler", "EvaluateModelSensitivity"]
            else:
                category = OutlierCategory.UNCERTAIN
                sev = EDASeverity.MINOR
                diag = f"Isolated statistical outliers ({iqr_outliers} rows, {iqr_ratio*100:.1f}%)."
                strategies = ["KeepUntouched", "RobustScaler"]

            findings.append(
                EDAFinding(
                    finding_id=f"eda_outlier_{col}",
                    category="outlier",
                    feature_name=col,
                    observation=f"Feature '{col}' has {iqr_outliers} IQR outliers ({iqr_ratio*100:.1f}%). Classified as {category.value}.",
                    evidence={
                        "iqr_outlier_count": iqr_outliers,
                        "extreme_3x_outliers": extreme_outliers,
                        "mad_outliers": mad_outliers,
                        "outlier_ratio": round(iqr_ratio, 4),
                        "classification": category.value,
                        "diagnosis": diag,
                    },
                    ml_impact="Extreme values heavily pull regression coefficients and OLS loss. May distort cluster centroids and PCA components.",
                    severity=sev,
                    suggested_investigation=diag,
                    candidate_strategies=strategies,
                    requires_validation=(category in (OutlierCategory.LIKELY_INVALID, OutlierCategory.POTENTIALLY_INVALID)),
                )
            )

        return findings
