"""Inter-dataset and intra-dataset relationship discovery engine."""

from typing import Dict, List
import pandas as pd
from amea.query_analysis.schemas import DatasetProfile, RelationshipItem


class RelationshipAnalyzer:
    """Discovers shared keys, schema matches, and strong numerical correlations."""

    @classmethod
    def analyze_relationships(
        cls,
        dfs: Dict[str, pd.DataFrame],
        profiles: List[DatasetProfile],
    ) -> List[RelationshipItem]:
        relationships: List[RelationshipItem] = []

        # 1. Intra-dataset numeric correlations
        for prof in profiles:
            df = dfs.get(prof.dataset_id)
            if df is None or df.empty:
                continue

            num_cols = prof.numeric_columns
            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                for i in range(len(num_cols)):
                    for j in range(i + 1, len(num_cols)):
                        c1, c2 = num_cols[i], num_cols[j]
                        val = float(corr.loc[c1, c2])
                        if abs(val) >= 0.5:
                            relationships.append(RelationshipItem(
                                source_a=prof.original_filename,
                                column_a=c1,
                                source_b=prof.original_filename,
                                column_b=c2,
                                relationship_type="correlation",
                                strength=abs(val),
                                evidence=f"Intra-dataset Pearson correlation r = {val:.2f} between '{c1}' and '{c2}'.",
                            ))

        # 2. Inter-dataset shared identifier / join key discovery
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                prof_a, prof_b = profiles[i], profiles[j]
                df_a, df_b = dfs.get(prof_a.dataset_id), dfs.get(prof_b.dataset_id)

                if df_a is None or df_b is None:
                    continue

                # Check column name overlap
                shared_cols = set(prof_a.column_names).intersection(set(prof_b.column_names))
                for col in shared_cols:
                    vals_a = set(df_a[col].dropna().unique())
                    vals_b = set(df_b[col].dropna().unique())

                    if vals_a and vals_b:
                        overlap = len(vals_a.intersection(vals_b))
                        union = len(vals_a.union(vals_b))
                        jaccard = overlap / union if union > 0 else 0.0

                        if jaccard > 0.2:
                            relationships.append(RelationshipItem(
                                source_a=prof_a.original_filename,
                                column_a=col,
                                source_b=prof_b.original_filename,
                                column_b=col,
                                relationship_type="shared_identifier",
                                strength=float(jaccard),
                                evidence=f"Shared identifier column '{col}' with {overlap} matching unique values (Jaccard = {jaccard:.2f}).",
                            ))

        return relationships
