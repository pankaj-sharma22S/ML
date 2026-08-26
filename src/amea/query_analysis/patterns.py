"""Statistical pattern discovery engine."""

from typing import Dict, List
import numpy as np
import pandas as pd
from amea.query_analysis.schemas import DatasetProfile, PatternItem, QueryIntent


class PatternDetector:
    """Discovers trends, concentrations, correlations, and outliers relevant to query."""

    @classmethod
    def detect_patterns(
        cls,
        dfs: Dict[str, pd.DataFrame],
        profiles: List[DatasetProfile],
        intent: QueryIntent,
    ) -> List[PatternItem]:
        patterns: List[PatternItem] = []

        for prof in profiles:
            df = dfs.get(prof.dataset_id)
            if df is None or df.empty:
                continue

            num_cols = prof.numeric_columns
            cat_cols = prof.categorical_columns

            # 1. Outlier Patterns
            for num_c in num_cols:
                series = df[num_c].dropna()
                if len(series) >= 10:
                    q1 = float(series.quantile(0.25))
                    q3 = float(series.quantile(0.75))
                    iqr = q3 - q1
                    if iqr > 0:
                        outliers = series[(series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))]
                        out_count = len(outliers)
                        if out_count > 0:
                            pct = (out_count / len(series)) * 100.0
                            patterns.append(PatternItem(
                                pattern_type="outlier",
                                description=f"Detected {out_count} outlier values in '{num_c}' ({pct:.1f}% of observations).",
                                strength=float(min(pct / 10.0, 1.0)),
                                evidence={
                                    "column": num_c,
                                    "outlier_count": out_count,
                                    "iqr_lower_bound": round(q1 - 1.5 * iqr, 2),
                                    "iqr_upper_bound": round(q3 + 1.5 * iqr, 2),
                                },
                            ))

            # 2. Categorical Concentration (Pareto Dominance)
            for cat_c in cat_cols:
                counts = df[cat_c].value_counts(normalize=True)
                if not counts.empty and float(counts.iloc[0]) >= 0.50:
                    top_cat = str(counts.index[0])
                    top_share = float(counts.iloc[0]) * 100.0
                    patterns.append(PatternItem(
                        pattern_type="dominance",
                        description=f"High category concentration: '{top_cat}' represents {top_share:.1f}% of all '{cat_c}' records.",
                        strength=float(top_share / 100.0),
                        evidence={"column": cat_c, "dominant_category": top_cat, "share_pct": round(top_share, 2)},
                    ))

        return patterns
