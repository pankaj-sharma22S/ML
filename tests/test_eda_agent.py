"""Unit and integration tests for EDA & Data Insight Agent."""

import numpy as np
import pandas as pd
import pytest

from amea.eda.agent import EDAAgent
from amea.eda.distribution_analyzer import DistributionAnalyzer
from amea.eda.outlier_analyzer import OutlierAnalyzer
from amea.eda.categorical_analyzer import CategoricalAnalyzer
from amea.eda.target_analyzer import TargetAnalyzer
from amea.eda.temporal_analyzer import TemporalAnalyzer
from amea.eda.relationship_analyzer import RelationshipAnalyzer
from amea.eda.models import EDASeverity, OutlierCategory


def test_distribution_analyzer():
    np.random.seed(42)
    # Right-skewed distribution
    skewed_data = np.random.exponential(scale=10.0, size=200)
    # Zero-inflated data
    zero_data = np.concatenate([np.zeros(50), np.random.normal(10, 2, 150)])

    df = pd.DataFrame({
        "skewed_feat": skewed_data,
        "zero_feat": zero_data,
        "normal_feat": np.random.normal(5, 1, 200),
    })

    distributions, findings = DistributionAnalyzer.analyze(df)
    assert len(distributions) == 3

    skew_dist = next(d for d in distributions if d.column_name == "skewed_feat")
    assert skew_dist.skewness > 1.0
    assert skew_dist.distribution_shape == "right_skewed"

    zero_dist = next(d for d in distributions if d.column_name == "zero_feat")
    assert zero_dist.is_zero_inflated
    assert zero_dist.zero_ratio >= 0.20

    assert len(findings) >= 2
    assert any("skewed_feat" in f.feature_name for f in findings)


def test_outlier_analyzer():
    np.random.seed(42)
    normal_base = np.random.normal(100, 10, 200)
    heavy_tail = np.concatenate([normal_base, [300.0, 350.0, 400.0, 450.0, 500.0]])
    ages = np.concatenate([np.random.randint(20, 60, size=203), [-10, 200]])

    df = pd.DataFrame({
        "age_with_invalid": ages,
        "salary_heavy_tail": heavy_tail,
    })

    findings = OutlierAnalyzer.analyze(df)
    assert len(findings) >= 2

    # Check likely invalid classification
    age_finding = next(f for f in findings if f.feature_name == "age_with_invalid")
    assert age_finding.evidence["classification"] == OutlierCategory.LIKELY_INVALID.value
    assert age_finding.severity == EDASeverity.CRITICAL
    assert age_finding.requires_validation

    # Check legitimate extreme classification
    salary_finding = next(f for f in findings if f.feature_name == "salary_heavy_tail")
    assert salary_finding.evidence["classification"] in (OutlierCategory.LEGITIMATE_EXTREME.value, OutlierCategory.POTENTIALLY_INVALID.value)


def test_categorical_analyzer():
    df = pd.DataFrame({
        "dominant_cat": ["A"] * 98 + ["B", "C"],  # 98% dominant
        "high_card": [f"ID_VAL_{i}" for i in range(100)],  # 100 distinct values
        "normal_cat": ["CAT_X", "CAT_Y"] * 50,
    })

    categoricals, findings = CategoricalAnalyzer.analyze(df)
    assert len(categoricals) == 3

    dom_cat = next(c for c in categoricals if c.column_name == "dominant_cat")
    assert dom_cat.dominant_ratio >= 0.95

    high_card = next(c for c in categoricals if c.column_name == "high_card")
    assert high_card.has_high_cardinality

    assert len(findings) >= 2
    assert any("dominant_cat" in f.feature_name for f in findings)
    assert any("high_card" in f.feature_name for f in findings)


def test_target_analyzer_imbalance():
    df = pd.DataFrame({
        "binary_target": [0] * 95 + [1] * 5,  # 95:5 (19:1) imbalance
        "continuous_target": np.random.exponential(scale=5.0, size=100),
    })

    # Test classification
    target_finding, findings = TargetAnalyzer.analyze(df, target_column="binary_target", is_classification=True)
    assert target_finding is not None
    assert target_finding.is_imbalanced
    assert target_finding.imbalance_ratio >= 10.0
    assert len(findings) == 1
    assert findings[0].severity == EDASeverity.CRITICAL

    # Test regression
    reg_finding, reg_findings = TargetAnalyzer.analyze(df, target_column="continuous_target", is_classification=False)
    assert reg_finding is not None
    assert reg_finding.task_type == "regression"


def test_temporal_analyzer():
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "feature_val": np.random.randn(100),
    })

    temporal_finding, findings = TemporalAnalyzer.analyze(df)
    assert temporal_finding is not None
    assert temporal_finding.is_monotonic_increasing
    assert temporal_finding.split_hazard_warning
    assert len(findings) == 1
    assert findings[0].severity == EDASeverity.CRITICAL
    assert "TimeSeriesSplitValidation" in findings[0].candidate_strategies


def test_relationship_analyzer():
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    df = pd.DataFrame({
        "f_a": x,
        "f_b": x * 3.0 - 2.0,  # Collinear
        "target": x * 1.5 + np.random.normal(0, 3.0, 100),  # Moderately correlated with target
    })

    findings = RelationshipAnalyzer.analyze(df, target_column="target", is_classification=False)
    assert len(findings) >= 2
    assert any("f_a <-> f_b" in f.feature_name for f in findings)
    assert any("f_a" in f.feature_name and "strong linear association" in f.observation.lower() for f in findings)


def test_eda_agent_end_to_end():
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65] * 10,
        "income": np.random.exponential(scale=50000, size=100),
        "department": ["Sales"] * 90 + ["HR"] * 10,
        "created_date": pd.date_range("2024-01-01", periods=100, freq="D"),
        "churn": [0] * 90 + [1] * 10,
    })

    agent = EDAAgent()
    report = agent.analyze(
        df=df,
        dataset_name="customer_churn_eda",
        target_column="churn",
        is_classification=True,
    )

    assert report.total_rows == 100
    assert report.total_columns == 5
    assert len(report.distributions) >= 2
    assert len(report.categoricals) >= 1
    assert report.target_analysis is not None
    assert report.target_analysis.is_imbalanced
    assert report.temporal_analysis is not None
    assert len(report.findings) > 0
