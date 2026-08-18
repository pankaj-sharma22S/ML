"""Feature engineering hypothesis generator backed by EDA and Data Intelligence findings."""

from typing import List
from amea.ml_strategy.models import FeatureEngineeringHypothesis


class FeatureHypothesizer:
    """Extracts testable feature engineering hypotheses from EDA findings and data distributions."""

    @staticmethod
    def generate_hypotheses(eda_findings: List[str]) -> List[FeatureEngineeringHypothesis]:
        hypotheses: List[FeatureEngineeringHypothesis] = []

        # 1. Check for Skewness findings
        skewed_findings = [f for f in eda_findings if "skew" in f.lower() or "tail" in f.lower()]
        if skewed_findings:
            hypotheses.append(
                FeatureEngineeringHypothesis(
                    hypothesis_id="hyp_power_transform",
                    transformation_name="PowerTransformer_YeoJohnson",
                    target_features=[],
                    expected_benefit="Stabilizes variance and normalizes heavy-tailed features, improving linear model convergence.",
                    risk_factor="Low risk; non-linear monotonic transformation.",
                    validation_method="Cross-validation comparison against standard scaled baseline.",
                    priority=1,
                )
            )

        # 2. Check for High Cardinality findings
        cardinality_findings = [f for f in eda_findings if "cardinality" in f.lower() or "rare" in f.lower()]
        if cardinality_findings:
            hypotheses.append(
                FeatureEngineeringHypothesis(
                    hypothesis_id="hyp_target_encoding",
                    transformation_name="TargetEncodingWithCrossFitting",
                    target_features=[],
                    expected_benefit="Compresses high-cardinality discrete categories into a 1D scalar signal without one-hot explosion.",
                    risk_factor="Potential train/test leakage if not strictly cross-fitted within folds.",
                    validation_method="Out-of-fold target encoding strictly computed on training split.",
                    priority=2,
                )
            )

        # 3. Check for Zero-Inflation findings
        zero_findings = [f for f in eda_findings if "zero" in f.lower()]
        if zero_findings:
            hypotheses.append(
                FeatureEngineeringHypothesis(
                    hypothesis_id="hyp_zero_indicator",
                    transformation_name="MissingIndicatorBinaryFeature",
                    target_features=[],
                    expected_benefit="Explicitly captures structural zero states as discrete categorical signals.",
                    risk_factor="Negligible risk.",
                    validation_method="Feature importance evaluation on tree models.",
                    priority=3,
                )
            )

        # Default interaction hypothesis
        if not hypotheses:
            hypotheses.append(
                FeatureEngineeringHypothesis(
                    hypothesis_id="hyp_standard_scaling",
                    transformation_name="StandardScaler_MedianImputer",
                    target_features=[],
                    expected_benefit="Provides standardized zero-mean unit-variance scaling for convex optimizers.",
                    risk_factor="None.",
                    validation_method="K-Fold cross-validation.",
                    priority=1,
                )
            )

        return hypotheses
