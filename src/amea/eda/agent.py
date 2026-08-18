"""EDA & Data Insight Agent coordinating multi-dimensional exploratory analysis."""

from pathlib import Path
from typing import Optional
import pandas as pd

from amea.eda.distribution_analyzer import DistributionAnalyzer
from amea.eda.outlier_analyzer import OutlierAnalyzer
from amea.eda.categorical_analyzer import CategoricalAnalyzer
from amea.eda.target_analyzer import TargetAnalyzer
from amea.eda.temporal_analyzer import TemporalAnalyzer
from amea.eda.relationship_analyzer import RelationshipAnalyzer
from amea.eda.models import EDAReport


class EDAAgent:
    """Diagnostic agent performing deep, evidence-driven exploratory data analysis."""

    def __init__(self):
        self.dist_analyzer = DistributionAnalyzer()
        self.outlier_analyzer = OutlierAnalyzer()
        self.cat_analyzer = CategoricalAnalyzer()
        self.target_analyzer = TargetAnalyzer()
        self.temp_analyzer = TemporalAnalyzer()
        self.rel_analyzer = RelationshipAnalyzer()

    def analyze(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        target_column: Optional[str] = None,
        is_classification: bool = True,
    ) -> EDAReport:
        """Execute complete EDA diagnostic pipeline and return structured EDAReport."""
        all_findings = []

        # 1. Distribution Analysis
        distributions, dist_findings = self.dist_analyzer.analyze(df=df)
        all_findings.extend(dist_findings)

        # 2. Outlier Analysis
        outlier_findings = self.outlier_analyzer.analyze(df=df)
        all_findings.extend(outlier_findings)

        # 3. Categorical Analysis
        categoricals, cat_findings = self.cat_analyzer.analyze(df=df)
        all_findings.extend(cat_findings)

        # 4. Target Analysis
        target_analysis, target_findings = self.target_analyzer.analyze(
            df=df,
            target_column=target_column,
            is_classification=is_classification,
        )
        all_findings.extend(target_findings)

        # 5. Temporal Analysis
        temporal_analysis, temp_findings = self.temp_analyzer.analyze(df=df)
        all_findings.extend(temp_findings)

        # 6. Relationship Analysis
        rel_findings = self.rel_analyzer.analyze(
            df=df,
            target_column=target_column,
            is_classification=is_classification,
        )
        all_findings.extend(rel_findings)

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]

        return EDAReport(
            dataset_name=dataset_name,
            total_rows=len(df),
            total_columns=len(df.columns),
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            findings=all_findings,
            distributions=distributions,
            categoricals=categoricals,
            target_analysis=target_analysis,
            temporal_analysis=temporal_analysis,
        )
