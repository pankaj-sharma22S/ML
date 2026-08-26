"""Query-driven selective visualization engine using Matplotlib and Seaborn."""

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from uuid import uuid4
import pandas as pd

from amea.query_analysis.artifact_manager import AnalysisArtifactManager
from amea.query_analysis.schemas import DatasetProfile, QueryIntent, VisualizationArtifact


class QueryDrivenVisualizer:
    """Generates charts strictly relevant to the analytical query intent."""

    def __init__(self, artifact_manager: Optional[AnalysisArtifactManager] = None):
        self.artifact_manager = artifact_manager or AnalysisArtifactManager()

    def generate_visualizations(
        self,
        dfs: Dict[str, pd.DataFrame],
        profiles: List[DatasetProfile],
        intent: QueryIntent,
        run_id: str,
    ) -> List[VisualizationArtifact]:
        artifacts: List[VisualizationArtifact] = []

        for prof in profiles:
            df = dfs.get(prof.dataset_id)
            if df is None or df.empty:
                continue

            num_cols = prof.numeric_columns
            cat_cols = prof.categorical_columns
            dt_cols = prof.datetime_columns

            # 1. Trend Analysis Line Chart
            if intent.primary_intent == "trend_analysis" or "trend_analysis" in intent.secondary_intents:
                metric_col = next((c for c in num_cols if any(m in c.lower() for m in intent.target_metrics)), num_cols[0] if num_cols else None)
                date_col = dt_cols[0] if dt_cols else None

                if metric_col and date_col:
                    temp_df = df.copy()
                    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
                    temp_df = temp_df.dropna(subset=[date_col]).sort_values(by=date_col)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(temp_df[date_col], temp_df[metric_col], marker="o", color="#2563eb", linewidth=2)
                    ax.set_title(f"Time Series Trend: {metric_col} over {date_col}", fontsize=12, fontweight="bold")
                    ax.set_xlabel(date_col)
                    ax.set_ylabel(metric_col)
                    ax.grid(True, linestyle="--", alpha=0.5)

                    path = self.artifact_manager.save_figure(fig, run_id, f"trend_{metric_col}")
                    artifacts.append(VisualizationArtifact(
                        id=f"viz_{uuid4().hex[:6]}",
                        chart_type="line_chart",
                        title=f"{metric_col} Trend Analysis",
                        reason=f"Generated line chart to visualize temporal trends in '{metric_col}'.",
                        artifact_path=path,
                        columns_visualized=[date_col, metric_col],
                    ))

            # 2. Ranking / Aggregation Bar Chart
            if intent.primary_intent in ["ranking", "aggregation"] or "ranking" in intent.secondary_intents:
                metric_col = next((c for c in num_cols if any(m in c.lower() for m in intent.target_metrics)), num_cols[0] if num_cols else None)
                dim_col = next((c for c in cat_cols if any(d in c.lower() for d in intent.target_dimensions)), cat_cols[0] if cat_cols else None)

                if metric_col and dim_col:
                    grouped = df.groupby(dim_col)[metric_col].sum().sort_values(ascending=False).head(10)
                    if not grouped.empty:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.barplot(x=grouped.values, y=grouped.index, ax=ax, palette="Blues_r")
                        ax.set_title(f"Top Contributors: {metric_col} by {dim_col}", fontsize=12, fontweight="bold")
                        ax.set_xlabel(f"Total {metric_col}")
                        ax.set_ylabel(dim_col)
                        ax.grid(True, linestyle="--", alpha=0.5, axis="x")

                        path = self.artifact_manager.save_figure(fig, run_id, f"ranking_{dim_col}")
                        artifacts.append(VisualizationArtifact(
                            id=f"viz_{uuid4().hex[:6]}",
                            chart_type="bar_chart",
                            title=f"Ranked {metric_col} by {dim_col}",
                            reason=f"Ranked bar chart highlighting leading drivers of '{metric_col}'.",
                            artifact_path=path,
                            columns_visualized=[dim_col, metric_col],
                        ))

            # 3. Correlation Heatmap
            if intent.primary_intent == "correlation" or "correlation" in intent.secondary_intents:
                if len(num_cols) >= 2:
                    corr = df[num_cols].corr()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
                    ax.set_title(f"Correlation Matrix: {prof.original_filename}", fontsize=12, fontweight="bold")

                    path = self.artifact_manager.save_figure(fig, run_id, "correlation_heatmap")
                    artifacts.append(VisualizationArtifact(
                        id=f"viz_{uuid4().hex[:6]}",
                        chart_type="correlation_heatmap",
                        title="Numeric Variables Correlation Heatmap",
                        reason="Heatmap showing pairwise linear correlations between numeric features.",
                        artifact_path=path,
                        columns_visualized=num_cols,
                    ))

            # 4. Outlier / Distribution Box Plot
            if intent.primary_intent in ["distribution", "anomaly_detection"] or "anomaly_detection" in intent.secondary_intents:
                metric_col = next((c for c in num_cols if any(m in c.lower() for m in intent.target_metrics)), num_cols[0] if num_cols else None)
                if metric_col:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.boxplot(x=df[metric_col], ax=ax, color="#38bdf8")
                    ax.set_title(f"Distribution & Outlier Plot: {metric_col}", fontsize=12, fontweight="bold")
                    ax.set_xlabel(metric_col)
                    ax.grid(True, linestyle="--", alpha=0.5)

                    path = self.artifact_manager.save_figure(fig, run_id, f"distribution_{metric_col}")
                    artifacts.append(VisualizationArtifact(
                        id=f"viz_{uuid4().hex[:6]}",
                        chart_type="box_plot",
                        title=f"{metric_col} Outlier & Distribution Box Plot",
                        reason=f"Visualizes spread, quartiles, and extreme values in '{metric_col}'.",
                        artifact_path=path,
                        columns_visualized=[metric_col],
                    ))

        return artifacts
