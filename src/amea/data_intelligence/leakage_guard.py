"""Target Leakage Guard detecting perfect predictors, ID artifacts, and contamination."""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from amea.data_intelligence.models import LeakageFinding, LeakageRiskLevel


class LeakageGuard:
    """Audits features for target leakage and artifact memorization."""

    @staticmethod
    def audit_leakage(
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        is_classification: bool = True,
    ) -> List[LeakageFinding]:
        findings: List[LeakageFinding] = []
        total_rows = len(df)

        if not target_col or target_col not in df.columns:
            return findings

        target = df[target_col]
        features = df.drop(columns=[target_col])

        # Prepare target numeric encoding if categorical
        if not pd.api.types.is_numeric_dtype(target):
            y_encoded = pd.factorize(target)[0]
        else:
            y_encoded = target.fillna(target.median()).values

        for col in features.columns:
            series = features[col]
            distinct_count = int(series.nunique(dropna=True))

            # 1. Identifier / Memorization Risk
            is_id_candidate = (
                distinct_count == total_rows
                and total_rows > 10
                and ("id" in col.lower() or "key" in col.lower() or pd.api.types.is_integer_dtype(series))
            )
            if is_id_candidate:
                findings.append(
                    LeakageFinding(
                        column_name=col,
                        risk_level=LeakageRiskLevel.CRITICAL,
                        reason="Column appears to be a unique row identifier or primary key.",
                        is_identifier_candidate=True,
                        recommended_action="Drop column before feature engineering and training.",
                    )
                )
                continue

            # 2. Perfect Predictor / Target Mirroring
            if pd.api.types.is_numeric_dtype(series):
                s_vals = series.fillna(series.median()).values
                if len(s_vals) > 5 and np.std(s_vals) > 1e-9 and np.std(y_encoded) > 1e-9:
                    corr = float(abs(np.corrcoef(s_vals, y_encoded)[0, 1]))
                    if not np.isnan(corr) and corr > 0.999:
                        findings.append(
                            LeakageFinding(
                                column_name=col,
                                risk_level=LeakageRiskLevel.CRITICAL,
                                reason=f"Feature exhibits near-perfect linear correlation (|R| = {corr:.4f}) with target variable.",
                                target_correlation=round(corr, 4),
                                recommended_action="High probability of synthetic or target-derived leakage. Exclude from training.",
                            )
                        )
                    elif not np.isnan(corr) and corr > 0.90:
                        findings.append(
                            LeakageFinding(
                                column_name=col,
                                risk_level=LeakageRiskLevel.HIGH,
                                reason=f"Very strong target correlation (|R| = {corr:.4f}). Audit business logic for post-outcome timing.",
                                target_correlation=round(corr, 4),
                                recommended_action="Verify timestamp and feature availability at inference time.",
                            )
                        )

        return findings
