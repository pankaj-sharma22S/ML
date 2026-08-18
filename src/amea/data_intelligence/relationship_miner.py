"""Relationship Miner discovering inter-feature correlations and associations."""

from typing import List
import numpy as np
import pandas as pd
from amea.data_intelligence.models import RelationshipFinding


class RelationshipMiner:
    """Discovers collinear clusters and non-linear associations between features."""

    @staticmethod
    def mine_relationships(df: pd.DataFrame, threshold: float = 0.85) -> List[RelationshipFinding]:
        findings: List[RelationshipFinding] = []
        numeric_df = df.select_dtypes(include=[np.number])
        cols = list(numeric_df.columns)

        if len(cols) < 2:
            return findings

        corr_matrix = numeric_df.corr().abs()

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                val = float(corr_matrix.loc[c1, c2])
                if not np.isnan(val) and val >= threshold:
                    findings.append(
                        RelationshipFinding(
                            feature_a=c1,
                            feature_b=c2,
                            relationship_type="linear_collinear",
                            strength=round(val, 4),
                            description=f"Strong linear collinearity (|R| = {val:.2f}) between '{c1}' and '{c2}'. Consider PCA, tree models, or pruning.",
                        )
                    )

        return findings
