"""Relationship and interaction analyzer diagnosing multicollinearity and predictive signals."""

from typing import List, Optional
import numpy as np
import pandas as pd

from amea.eda.models import EDAFinding, EDASeverity


class RelationshipAnalyzer:
    """Investigates inter-feature correlations, collinearity clusters, and predictive interactions."""

    @staticmethod
    def analyze(
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        is_classification: bool = True,
    ) -> List[EDAFinding]:
        findings: List[EDAFinding] = []
        numeric_df = df.select_dtypes(include=[np.number])
        cols = list(numeric_df.columns)

        if len(cols) < 2:
            return findings

        corr_matrix = numeric_df.corr().abs()

        # 1. High Collinearity Clusters
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                if c1 == target_column or c2 == target_column:
                    continue

                r_val = float(corr_matrix.loc[c1, c2])
                if not np.isnan(r_val) and r_val >= 0.90:
                    findings.append(
                        EDAFinding(
                            finding_id=f"eda_collinear_{c1}_{c2}",
                            category="relationship",
                            feature_name=f"{c1} <-> {c2}",
                            observation=f"Near-perfect collinearity (|R| = {r_val:.3f}) between features '{c1}' and '{c2}'.",
                            evidence={"feature_a": c1, "feature_b": c2, "correlation": r_val},
                            ml_impact="Inflates variance of linear model coefficients, causes multicollinearity instability in regularized models.",
                            severity=EDASeverity.IMPORTANT,
                            suggested_investigation="Drop one feature of the pair, apply PCA/SVD dimensionality reduction, or use tree ensembles.",
                            candidate_strategies=["DropCollinearFeature", "PCA_DimensionalityReduction", "L1_LassoFeatureSelection"],
                            requires_validation=False,
                        )
                    )

        # 2. Strong Predictive Signal with Target
        if target_column and target_column in numeric_df.columns:
            for col in cols:
                if col == target_column:
                    continue
                r_target = float(corr_matrix.loc[col, target_column])
                if not np.isnan(r_target) and 0.30 <= r_target < 0.999:
                    findings.append(
                        EDAFinding(
                            finding_id=f"eda_strong_predictor_{col}",
                            category="relationship",
                            feature_name=col,
                            observation=f"Feature '{col}' exhibits strong linear association (|R| = {r_target:.3f}) with target '{target_column}'.",
                            evidence={"feature": col, "target": target_column, "correlation": r_target},
                            ml_impact="High predictive power. Key candidate for linear and tree models.",
                            severity=EDASeverity.INFORMATIONAL,
                            suggested_investigation="Retain feature and evaluate interaction terms with other moderately correlated features.",
                            candidate_strategies=["RetainCoreFeature", "ExploreFeatureInteractions"],
                            requires_validation=False,
                        )
                    )

        return findings
