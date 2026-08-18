"""Target variable analyzer diagnosing class imbalance, skew, and validation design."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from amea.eda.models import EDAFinding, EDASeverity, TargetAnalysisFinding


class TargetAnalyzer:
    """Performs deep target-centric diagnostics to inform loss functions and CV schemes."""

    @staticmethod
    def analyze(
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        is_classification: bool = True,
    ) -> Tuple[Optional[TargetAnalysisFinding], List[EDAFinding]]:
        findings: List[EDAFinding] = []

        if not target_column or target_column not in df.columns:
            return None, findings

        target = df[target_column].dropna()
        total_rows = len(target)
        if total_rows == 0:
            return None, findings

        if is_classification:
            # Classification Target Analysis
            val_counts = target.value_counts(normalize=True)
            counts = target.value_counts()
            class_dist = {str(k): round(float(v), 4) for k, v in val_counts.items()}

            majority_ratio = float(val_counts.iloc[0])
            minority_ratio = float(val_counts.iloc[-1])
            minority_count = int(counts.iloc[-1])
            imbalance_ratio = round(majority_ratio / max(1e-6, minority_ratio), 2)
            is_imbalanced = (imbalance_ratio >= 3.0 or minority_ratio <= 0.15)

            target_finding = TargetAnalysisFinding(
                target_column=target_column,
                task_type="classification",
                is_imbalanced=is_imbalanced,
                imbalance_ratio=imbalance_ratio,
                class_distribution=class_dist,
                minority_class_count=minority_count,
                target_summary=f"Classification target with {len(val_counts)} classes. Imbalance ratio: {imbalance_ratio}:1.",
            )

            if is_imbalanced:
                sev = EDASeverity.CRITICAL if imbalance_ratio >= 10.0 else EDASeverity.IMPORTANT
                findings.append(
                    EDAFinding(
                        finding_id=f"eda_target_imbalance_{target_column}",
                        category="target",
                        feature_name=target_column,
                        observation=f"Target '{target_column}' is significantly imbalanced (Imbalance ratio = {imbalance_ratio}:1, minority class has {minority_count} samples / {minority_ratio*100:.1f}%).",
                        evidence={"imbalance_ratio": imbalance_ratio, "class_distribution": class_dist, "minority_count": minority_count},
                        ml_impact="Standard Accuracy metric is deceptive. Models will favor the majority class without penalty. Non-stratified CV may yield empty minority folds.",
                        severity=sev,
                        suggested_investigation="Enforce StratifiedKFold cross-validation, optimize PR-AUC/F1-Macro over ROC-AUC, and evaluate balanced class weighting or focal loss.",
                        candidate_strategies=["StratifiedKFoldValidation", "BalancedClassWeights", "OptimizePRAUC", "ThresholdTuning", "FocalLoss"],
                        requires_validation=True,
                    )
                )

        else:
            # Regression Target Analysis
            vals = target.values
            mean_val = float(np.mean(vals))
            median_val = float(np.median(vals))
            std_val = float(np.std(vals))
            skew_val = float(stats.skew(vals)) if std_val > 1e-9 else 0.0
            zero_count = int(np.sum(vals == 0))
            zero_ratio = float(zero_count / total_rows)

            target_finding = TargetAnalysisFinding(
                target_column=target_column,
                task_type="regression",
                skewness=round(skew_val, 4),
                is_zero_inflated=(zero_ratio >= 0.15),
                target_summary=f"Regression target: mean={mean_val:.2f}, median={median_val:.2f}, skewness={skew_val:.2f}.",
            )

            if abs(skew_val) > 2.0:
                findings.append(
                    EDAFinding(
                        finding_id=f"eda_target_skew_{target_column}",
                        category="target",
                        feature_name=target_column,
                        observation=f"Continuous target '{target_column}' is heavily skewed (skewness = {skew_val:.2f}, mean={mean_val:.2f} vs median={median_val:.2f}).",
                        evidence={"skewness": skew_val, "mean": mean_val, "median": median_val},
                        ml_impact="MSE / RMSE loss functions will excessively penalize errors on extreme positive outliers, hurting general prediction accuracy.",
                        severity=EDASeverity.IMPORTANT,
                        suggested_investigation="Evaluate log1p target transformation or TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1).",
                        candidate_strategies=["LogTransformTarget", "TransformedTargetRegressor", "HuberLoss", "MAE_Optimization"],
                        requires_validation=True,
                    )
                )

        return target_finding, findings
