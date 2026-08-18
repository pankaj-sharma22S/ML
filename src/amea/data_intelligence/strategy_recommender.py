"""Evidence-backed data treatment strategy recommender."""

from typing import List
from amea.data_intelligence.models import (
    DataTreatmentCandidate,
    QualityAuditReport,
    LeakageFinding,
    LeakageRiskLevel,
    RelationshipFinding,
)


class StrategyRecommender:
    """Synthesizes diagnostic findings into actionable candidate preprocessing strategies."""

    @staticmethod
    def recommend(
        quality_audit: QualityAuditReport,
        leakage_findings: List[LeakageFinding],
        relationships: List[RelationshipFinding],
    ) -> List[DataTreatmentCandidate]:
        candidates: List[DataTreatmentCandidate] = []

        # 1. Leakage / Identifier Dropping Strategy
        critical_leaks = [f.column_name for f in leakage_findings if f.risk_level == LeakageRiskLevel.CRITICAL]
        if critical_leaks:
            candidates.append(
                DataTreatmentCandidate(
                    strategy_id="strat_drop_leakage_identifiers",
                    target_columns=critical_leaks,
                    treatment_type="drop_feature",
                    proposed_transformer="ColumnDropper",
                    rationale=f"Columns {critical_leaks} identified as unique IDs or perfect target mirrors.",
                    pros=["Eliminates target leakage", "Prevents model memorization"],
                    cons=["Slightly reduces feature set dimension"],
                    expected_impact="High generalization improvement on unseen test data.",
                )
            )

        # 2. Imputation Strategies based on Missingness Mechanisms
        if quality_audit.missingness_findings:
            missing_cols = [m.column_name for m in quality_audit.missingness_findings if m.column_name not in critical_leaks]
            if missing_cols:
                candidates.append(
                    DataTreatmentCandidate(
                        strategy_id="strat_imputation_pipeline",
                        target_columns=missing_cols,
                        treatment_type="imputation",
                        proposed_transformer="SimpleImputer(strategy='median')",
                        rationale=f"Missing values detected in {missing_cols}.",
                        pros=["Preserves sample row count", "Computationally efficient"],
                        cons=["May reduce feature variance"],
                        expected_impact="Ensures all downstream models receive non-null feature matrices.",
                    )
                )

        # 3. Robust Scaling for Outlier / Skewed Features
        severe_outliers = [o.column_name for o in quality_audit.outlier_findings if o.is_severe and o.column_name not in critical_leaks]
        if severe_outliers:
            candidates.append(
                DataTreatmentCandidate(
                    strategy_id="strat_robust_scaling",
                    target_columns=severe_outliers,
                    treatment_type="scaling",
                    proposed_transformer="RobustScaler()",
                    rationale=f"Features {severe_outliers} exhibit heavy tails and severe outlier ratios.",
                    pros=["Reduces influence of extreme outliers on gradient and linear models", "Preserves distribution rank"],
                    cons=["Less standard than z-score standardization"],
                    expected_impact="Improves convergence and stability of linear/neural models.",
                )
            )

        # 4. Standard Scaling for Non-Skewed Numeric Features
        normal_numeric = [o.column_name for o in quality_audit.outlier_findings if not o.is_severe and o.column_name not in critical_leaks]
        if normal_numeric:
            candidates.append(
                DataTreatmentCandidate(
                    strategy_id="strat_standard_scaling",
                    target_columns=normal_numeric,
                    treatment_type="scaling",
                    proposed_transformer="StandardScaler()",
                    rationale="Standard variance normalization for well-behaved continuous features.",
                    pros=["Standardizes feature scale to zero mean unit variance"],
                    cons=["Sensitive to extreme unclipped outliers"],
                    expected_impact="Ensures equal feature weighting in distance/gradient models.",
                )
            )

        return candidates
