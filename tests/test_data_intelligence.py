"""Unit and integration tests for Data Intelligence & Data Quality Agent."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from amea.data_intelligence.agent import DataIntelligenceAgent
from amea.data_intelligence.lineage import DatasetLineageManager
from amea.data_intelligence.profiler import DeepDataProfiler
from amea.data_intelligence.quality_auditor import QualityAuditor
from amea.data_intelligence.leakage_guard import LeakageGuard
from amea.data_intelligence.relationship_miner import RelationshipMiner
from amea.data_intelligence.models import MissingnessMechanism, LeakageRiskLevel


def test_lineage_and_hashing(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    version_meta = DatasetLineageManager.create_version(source_path=csv_path, df=df)
    assert version_meta.source_hash_sha256 is not None
    assert len(version_meta.source_hash_sha256) == 64
    assert version_meta.schema_hash is not None


def test_deep_data_profiler(tmp_path):
    df = pd.DataFrame({
        "num_col": [1.0, 2.0, 3.0, 4.0, 100.0],
        "cat_col": ["A", "B", "A", "B", "C"],
        "const_col": [1, 1, 1, 1, 1],
    })
    csv_path = tmp_path / "profile_test.csv"
    df.to_csv(csv_path, index=False)

    profile = DeepDataProfiler.profile(df, str(csv_path), "fake_sha")
    assert profile.total_rows == 5
    assert profile.total_columns == 3
    assert profile.columns["const_col"].is_constant
    assert not profile.columns["num_col"].is_constant
    assert profile.columns["num_col"].mean == 22.0
    assert profile.columns["cat_col"].class_balance is not None


def test_quality_auditor_missingness_and_outliers():
    np.random.seed(42)
    df = pd.DataFrame({
        "complete": np.random.randn(100),
        "missing_mcar": [np.nan if i % 10 == 0 else 1.0 for i in range(100)],
        "missing_mnar": [np.nan if i < 60 else 2.0 for i in range(100)],
        "skewed_outliers": np.concatenate([np.random.normal(0, 1, 95), [50.0, 60.0, 70.0, 80.0, 90.0]]),
    })

    report = QualityAuditor.audit(df)
    assert not report.is_clean
    assert len(report.missingness_findings) == 2

    # Check MNAR detection for high missingness column
    mnar_finding = next(f for f in report.missingness_findings if f.column_name == "missing_mnar")
    assert mnar_finding.candidate_mechanism == MissingnessMechanism.MNAR

    # Check outlier detection
    outlier_finding = next(f for f in report.outlier_findings if f.column_name == "skewed_outliers")
    assert outlier_finding.iqr_outlier_count >= 5
    assert outlier_finding.is_severe


def test_leakage_guard_detection():
    df = pd.DataFrame({
        "id_col": list(range(100)),
        "target": [0, 1] * 50,
        "clean_feature": np.random.randn(100),
    })
    # Add a feature that perfectly mirrors target
    df["perfect_leak"] = df["target"].values

    findings = LeakageGuard.audit_leakage(df, target_col="target", is_classification=True)
    assert len(findings) >= 2

    # Check ID detection
    id_finding = next(f for f in findings if f.column_name == "id_col")
    assert id_finding.risk_level == LeakageRiskLevel.CRITICAL
    assert id_finding.is_identifier_candidate

    # Check perfect target mirror
    leak_finding = next(f for f in findings if f.column_name == "perfect_leak")
    assert leak_finding.risk_level == LeakageRiskLevel.CRITICAL
    assert leak_finding.target_correlation == 1.0


def test_relationship_miner():
    x = np.linspace(0, 10, 100)
    df = pd.DataFrame({
        "feat_1": x,
        "feat_2": x * 2.0 + 1.0,  # Perfectly collinear
        "feat_3": np.random.randn(100),
    })

    relationships = RelationshipMiner.mine_relationships(df, threshold=0.85)
    assert len(relationships) == 1
    assert relationships[0].feature_a == "feat_1"
    assert relationships[0].feature_b == "feat_2"
    assert relationships[0].strength > 0.99


def test_data_intelligence_agent_end_to_end(tmp_path):
    df = pd.DataFrame({
        "user_id": list(range(100)),
        "age": np.random.randint(18, 70, size=100),
        "income": np.random.exponential(scale=50000, size=100),
        "target_churn": [0, 1] * 50,
    })
    csv_path = tmp_path / "churn_raw.csv"
    df.to_csv(csv_path, index=False)

    agent = DataIntelligenceAgent()
    evidence_pkg, profile = agent.process_dataset(
        dataset_path=csv_path,
        target_column="target_churn",
        is_classification=True,
    )

    assert evidence_pkg.total_rows == 100
    assert evidence_pkg.total_columns == 4
    assert profile.total_rows == 100
    assert len(evidence_pkg.treatment_candidates) > 0
    assert len(evidence_pkg.summary_findings) > 0
    assert any("user_id" in f.column_name for f in evidence_pkg.leakage_findings)
