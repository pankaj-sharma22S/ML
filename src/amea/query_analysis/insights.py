"""Statistical, calculation-backed insight generation engine."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from amea.query_analysis.schemas import DatasetProfile, InsightItem, QueryIntent


class InsightGenerator:
    """Computes exact, data-backed analytical insights based on query intent."""

    @classmethod
    def generate_insights(
        cls,
        dfs: Dict[str, pd.DataFrame], # dataset_id -> DataFrame
        profiles: List[DatasetProfile],
        intent: QueryIntent,
    ) -> List[InsightItem]:
        insights: List[InsightItem] = []

        for prof in profiles:
            df = dfs.get(prof.dataset_id)
            if df is None or df.empty:
                continue

            num_cols = prof.numeric_columns
            cat_cols = prof.categorical_columns
            dt_cols = prof.datetime_columns

            # 1. Ranking & Contribution Insights
            if intent.primary_intent in ["ranking", "aggregation"] or "ranking" in intent.secondary_intents:
                # Find metric and dimension candidates
                metric_col = next((c for c in num_cols if any(m in c.lower() for m in intent.target_metrics)), num_cols[0] if num_cols else None)
                dim_col = next((c for c in cat_cols if any(d in c.lower() for d in intent.target_dimensions)), cat_cols[0] if cat_cols else None)

                if metric_col and dim_col:
                    grouped = df.groupby(dim_col)[metric_col].sum().sort_values(ascending=False)
                    total = float(grouped.sum())
                    if total > 0 and len(grouped) > 0:
                        top_dim = str(grouped.index[0])
                        top_val = float(grouped.iloc[0])
                        top_pct = (top_val / total) * 100.0

                        insights.append(InsightItem(
                            insight=f"'{top_dim}' is the leading driver of {metric_col}, accounting for {top_val:,.2f} ({top_pct:.1f}% of total).",
                            evidence=f"Aggregated {metric_col} across {len(grouped)} distinct '{dim_col}' categories. Total: {total:,.2f}.",
                            metric=metric_col,
                            dimension=dim_col,
                            calculation={
                                "top_category": top_dim,
                                "top_value": top_val,
                                "total": total,
                                "share_percentage": round(top_pct, 2),
                            },
                            affected_dataset=prof.dataset_id,
                            confidence=0.95,
                        ))

            # 2. Trend & Growth Insights
            if intent.primary_intent == "trend_analysis" or "trend_analysis" in intent.secondary_intents:
                metric_col = next((c for c in num_cols if any(m in c.lower() for m in intent.target_metrics)), num_cols[0] if num_cols else None)
                date_col = dt_cols[0] if dt_cols else None

                if metric_col and date_col:
                    temp_df = df.copy()
                    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
                    temp_df = temp_df.dropna(subset=[date_col]).sort_values(by=date_col)
                    if len(temp_df) >= 2:
                        first_val = float(temp_df.iloc[0][metric_col])
                        last_val = float(temp_df.iloc[-1][metric_col])
                        pct_change = ((last_val - first_val) / first_val * 100.0) if first_val != 0 else 0.0

                        direction = "increased" if pct_change > 0 else "declined"
                        insights.append(InsightItem(
                            insight=f"{metric_col} {direction} by {abs(pct_change):.1f}% from {temp_df.iloc[0][date_col].strftime('%Y-%m-%d')} ({first_val:,.2f}) to {temp_df.iloc[-1][date_col].strftime('%Y-%m-%d')} ({last_val:,.2f}).",
                            evidence=f"Time series analysis across {len(temp_df)} temporal data points.",
                            metric=metric_col,
                            dimension=date_col,
                            calculation={
                                "initial_value": first_val,
                                "final_value": last_val,
                                "percentage_change": round(pct_change, 2),
                            },
                            affected_dataset=prof.dataset_id,
                            confidence=0.92,
                        ))

            # 3. Correlation Insights
            if intent.primary_intent == "correlation" or "correlation" in intent.secondary_intents:
                if len(num_cols) >= 2:
                    corr_matrix = df[num_cols].corr()
                    # Find strongest non-trivial pair
                    best_pair = None
                    best_corr = 0.0
                    for i in range(len(num_cols)):
                        for j in range(i + 1, len(num_cols)):
                            c1, c2 = num_cols[i], num_cols[j]
                            val = corr_matrix.loc[c1, c2]
                            if not np.isnan(val) and abs(val) > abs(best_corr):
                                best_corr = float(val)
                                best_pair = (c1, c2)

                    if best_pair and abs(best_corr) >= 0.3:
                        c1, c2 = best_pair
                        strength = "strong" if abs(best_corr) > 0.7 else "moderate"
                        direction = "positive" if best_corr > 0 else "negative"
                        insights.append(InsightItem(
                            insight=f"Found a {strength} {direction} correlation (r = {best_corr:.2f}) between '{c1}' and '{c2}'.",
                            evidence=f"Pearson correlation calculated across {len(df)} records.",
                            metric=f"{c1} & {c2}",
                            calculation={"pearson_r": round(best_corr, 3), "columns": list(best_pair)},
                            affected_dataset=prof.dataset_id,
                            confidence=0.90,
                        ))

        return insights
