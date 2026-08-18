"""Categorical feature analysis diagnosing cardinality, dominance, and rare categories."""

from typing import List, Tuple
import pandas as pd

from amea.eda.models import CategoricalFinding, EDAFinding, EDASeverity


class CategoricalAnalyzer:
    """Analyzes discrete categorical features, frequency distributions, and cardinality."""

    @staticmethod
    def analyze(df: pd.DataFrame) -> Tuple[List[CategoricalFinding], List[EDAFinding]]:
        findings: List[EDAFinding] = []
        categoricals: List[CategoricalFinding] = []
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])]

        total_rows = len(df)
        if total_rows == 0:
            return categoricals, findings

        for col in cat_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            distinct_count = int(series.nunique())
            cardinality_ratio = float(distinct_count / total_rows)
            val_counts = series.value_counts(normalize=True)

            dominant_cat = str(val_counts.index[0]) if len(val_counts) > 0 else None
            dominant_ratio = float(val_counts.iloc[0]) if len(val_counts) > 0 else 0.0

            # Rare categories (< 1% frequency)
            rare_cats = val_counts[val_counts < 0.01]
            rare_count = len(rare_cats)
            rare_ratio = float(rare_cats.sum())

            has_high_cardinality = (distinct_count > 50 or (cardinality_ratio > 0.20 and distinct_count > 20))

            cat_finding = CategoricalFinding(
                column_name=col,
                distinct_count=distinct_count,
                cardinality_ratio=round(cardinality_ratio, 4),
                dominant_category=dominant_cat,
                dominant_ratio=round(dominant_ratio, 4),
                rare_categories_count=rare_count,
                rare_categories_ratio=round(rare_ratio, 4),
                has_high_cardinality=has_high_cardinality,
            )
            categoricals.append(cat_finding)

            # Generate actionable findings
            if dominant_ratio >= 0.95:
                findings.append(
                    EDAFinding(
                        finding_id=f"eda_cat_dominant_{col}",
                        category="categorical",
                        feature_name=col,
                        observation=f"Categorical feature '{col}' is heavily dominated by category '{dominant_cat}' ({dominant_ratio*100:.1f}% of values).",
                        evidence={"dominant_category": dominant_cat, "dominant_ratio": dominant_ratio},
                        ml_impact="Near-zero variance in categorical domain. Offers minimal predictive signal and may cause split instability.",
                        severity=EDASeverity.MINOR,
                        suggested_investigation="Evaluate whether column should be dropped or combined into a binary indicator.",
                        candidate_strategies=["BinaryIndicatorForNonDominant", "DropNearConstantFeature"],
                        requires_validation=False,
                    )
                )

            if has_high_cardinality:
                findings.append(
                    EDAFinding(
                        finding_id=f"eda_cat_high_cardinality_{col}",
                        category="categorical",
                        feature_name=col,
                        observation=f"Categorical feature '{col}' has high cardinality ({distinct_count} distinct categories in {total_rows} rows).",
                        evidence={"distinct_count": distinct_count, "cardinality_ratio": cardinality_ratio},
                        ml_impact="One-hot encoding will produce an excessively sparse, high-dimensional matrix. Risk of tree model overfitting.",
                        severity=EDASeverity.IMPORTANT,
                        suggested_investigation="Evaluate Target Encoding with cross-fitting, Frequency/Count Encoding, or Category Embedding.",
                        candidate_strategies=["TargetEncodingWithCrossFitting", "FrequencyEncoding", "GroupRareCategories", "CatBoostNativeEncoding"],
                        requires_validation=False,
                    )
                )

            if rare_count > 5:
                findings.append(
                    EDAFinding(
                        finding_id=f"eda_cat_rare_categories_{col}",
                        category="categorical",
                        feature_name=col,
                        observation=f"Feature '{col}' has {rare_count} rare categories with frequency < 1% (accounting for {rare_ratio*100:.1f}% total rows).",
                        evidence={"rare_count": rare_count, "rare_ratio": rare_ratio},
                        ml_impact="Rare categories may appear in test/validation folds without having been seen during training, causing missing level errors.",
                        severity=EDASeverity.MINOR,
                        suggested_investigation="Group rare categories into an 'OTHER' bin during preprocessing.",
                        candidate_strategies=["GroupRareCategoriesOther", "HandleUnknownCategoricals"],
                        requires_validation=False,
                    )
                )

        return categoricals, findings
