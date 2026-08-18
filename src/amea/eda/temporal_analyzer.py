"""Temporal analyzer diagnosing timestamps, monotonicity, and split hazard warnings."""

from typing import List, Optional, Tuple
import pandas as pd

from amea.eda.models import EDAFinding, EDASeverity, TemporalFinding


class TemporalAnalyzer:
    """Analyzes time-based features and enforces time-series validation boundaries."""

    @staticmethod
    def analyze(df: pd.DataFrame) -> Tuple[Optional[TemporalFinding], List[EDAFinding]]:
        findings: List[EDAFinding] = []
        datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in c.lower() or "time" in c.lower() or "timestamp" in c.lower()]

        if not datetime_cols:
            return None, findings

        primary_time_col = datetime_cols[0]
        series = df[primary_time_col]

        # Try to parse to datetime if string/object
        try:
            parsed_dt = pd.to_datetime(series, errors="coerce")
            valid_dt = parsed_dt.dropna()
            if len(valid_dt) < 5:
                return None, findings

            is_monotonic = valid_dt.is_monotonic_increasing
            span_days = float((valid_dt.max() - valid_dt.min()).total_seconds() / (24 * 3600))

            temporal_finding = TemporalFinding(
                datetime_column=primary_time_col,
                is_monotonic_increasing=is_monotonic,
                temporal_span_days=round(span_days, 2),
                split_hazard_warning=True,
                recommendation="Temporal column detected. Must use TimeSeriesSplit or PurgedGroupTimeSeriesSplit to avoid lookahead leakage.",
            )

            findings.append(
                EDAFinding(
                    finding_id=f"eda_temporal_split_hazard_{primary_time_col}",
                    category="temporal",
                    feature_name=primary_time_col,
                    observation=f"Dataset contains temporal column '{primary_time_col}' spanning {span_days:.1f} days. Random K-Fold splitting creates severe temporal lookahead leakage.",
                    evidence={"datetime_column": primary_time_col, "is_monotonic": is_monotonic, "span_days": span_days},
                    ml_impact="Random train/test splits allow future information to leak into past predictions, yielding overly optimistic validation scores that fail in production.",
                    severity=EDASeverity.CRITICAL,
                    suggested_investigation="Enforce chronological TimeSeriesSplit or expanding window cross-validation. Prohibit standard K-Fold shuffle.",
                    candidate_strategies=["TimeSeriesSplitValidation", "ChronologicalOrderEnforcement", "LaggedFeatureGeneration", "PurgedGroupSplit"],
                    requires_validation=True,
                )
            )

            return temporal_finding, findings

        except Exception:
            return None, findings
