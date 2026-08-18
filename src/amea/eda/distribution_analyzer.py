"""Numeric distribution analysis diagnosing skewness, kurtosis, and heavy tails."""

from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from amea.eda.models import DistributionFinding, EDAFinding, EDASeverity


class DistributionAnalyzer:
    """Analyzes continuous and discrete numeric distributions deterministically."""

    @staticmethod
    def analyze(df: pd.DataFrame) -> Tuple[List[DistributionFinding], List[EDAFinding]]:
        findings: List[EDAFinding] = []
        distributions: List[DistributionFinding] = []
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 5:
                continue

            vals = series.values
            mean_val = float(np.mean(vals))
            median_val = float(np.median(vals))
            std_val = float(np.std(vals))
            q25, q75 = np.percentile(vals, [25, 75])
            iqr_val = float(q75 - q25)

            skew_val = float(stats.skew(vals)) if std_val > 1e-9 else 0.0
            kurt_val = float(stats.kurtosis(vals)) if std_val > 1e-9 else 0.0

            zero_count = int(np.sum(vals == 0))
            zero_ratio = float(zero_count / len(vals))
            is_zero_inflated = (zero_ratio >= 0.20)
            is_heavy_tailed = (kurt_val > 3.0)

            # Classify shape
            if std_val <= 1e-9:
                shape = "constant"
            elif skew_val > 1.0:
                shape = "right_skewed"
            elif skew_val < -1.0:
                shape = "left_skewed"
            else:
                shape = "symmetric"

            dist_finding = DistributionFinding(
                column_name=col,
                mean=round(mean_val, 4),
                median=round(median_val, 4),
                std=round(std_val, 4),
                iqr=round(iqr_val, 4),
                skewness=round(skew_val, 4),
                kurtosis=round(kurt_val, 4),
                is_zero_inflated=is_zero_inflated,
                zero_ratio=round(zero_ratio, 4),
                is_heavy_tailed=is_heavy_tailed,
                distribution_shape=shape,
            )
            distributions.append(dist_finding)

            # Generate actionable EDA findings for anomalies
            if abs(skew_val) > 1.0:
                sev = EDASeverity.IMPORTANT if abs(skew_val) > 2.5 else EDASeverity.MINOR
                direction = "right-skewed (heavy positive tail)" if skew_val > 0 else "left-skewed (heavy negative tail)"
                findings.append(
                    EDAFinding(
                        finding_id=f"eda_dist_skew_{col}",
                        category="distribution",
                        feature_name=col,
                        observation=f"Feature '{col}' is strongly {direction} (skewness = {skew_val:.2f}, kurtosis = {kurt_val:.2f}).",
                        evidence={"mean": mean_val, "median": median_val, "skewness": skew_val, "kurtosis": kurt_val},
                        ml_impact="Linear and neural models may suffer gradient instability or sub-optimal convergence. Tree models are naturally robust.",
                        severity=sev,
                        suggested_investigation="Evaluate log1p, PowerTransformer (Yeo-Johnson), or QuantileTransformer vs tree-based models.",
                        candidate_strategies=["PowerTransform", "Log1pTransform", "RobustScaler", "PreferTreeModels"],
                        requires_validation=False,
                    )
                )

            if is_zero_inflated:
                findings.append(
                    EDAFinding(
                        finding_id=f"eda_dist_zero_inflated_{col}",
                        category="distribution",
                        feature_name=col,
                        observation=f"Feature '{col}' exhibits significant zero-inflation ({zero_ratio*100:.1f}% zeros).",
                        evidence={"zero_count": zero_count, "zero_ratio": zero_ratio},
                        ml_impact="Single standard scaling treats zeros as non-special continuous values, potentially diluting sparsity signals.",
                        severity=EDASeverity.INFORMATIONAL,
                        suggested_investigation="Consider creating a binary indicator feature (is_zero) alongside the continuous feature value.",
                        candidate_strategies=["BinaryIndicatorFeature", "TweedieRegressor", "HurdleModel"],
                        requires_validation=False,
                    )
                )

        return distributions, findings
