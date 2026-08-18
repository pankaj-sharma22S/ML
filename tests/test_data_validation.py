"""Unit and integration tests for Data Validation & Quality Gate Agent."""

import numpy as np
import pandas as pd
import pytest

from amea.data_validation.agent import DataValidationAgent
from amea.data_validation.schema_validator import SchemaValidator
from amea.data_validation.integrity_validator import IntegrityValidator
from amea.data_validation.distribution_auditor import DistributionAuditor
from amea.data_validation.domain_validator import DomainValidator
from amea.data_validation.models import CheckStatus, QualityGateVerdict


def test_schema_validator():
    raw_df = pd.DataFrame({"feat_1": [1, 2], "feat_2": [3, 4], "target": [0, 1]})
    clean_df = pd.DataFrame({"feat_1": [1, 2], "feat_2": [3, 4]})  # Missing target column!

    results = SchemaValidator.validate(raw_df, clean_df, target_column="target")
    assert any(r.check_name == "target_column_existence" and r.status == CheckStatus.FAIL for r in results)


def test_integrity_validator_null_detection():
    raw_df = pd.DataFrame({"feat": [1.0, 2.0], "target": [0, 1]})
    clean_df = pd.DataFrame({"feat": [1.0, np.nan], "target": [0, 1]})  # Residual null!

    results = IntegrityValidator.validate(raw_df, clean_df, target_column="target")
    assert any(r.check_name == "residual_missingness_check" and r.status == CheckStatus.FAIL for r in results)


def test_distribution_auditor_distortion():
    np.random.seed(42)
    raw_vals = np.random.normal(50, 5, 200)
    # Severely distorted cleaned values (mean shift + massive variance change)
    corrupted_vals = np.random.normal(100, 50, 200)

    raw_df = pd.DataFrame({"metric": raw_vals})
    clean_df = pd.DataFrame({"metric": corrupted_vals})

    shift_metrics, check_results = DistributionAuditor.audit(raw_df, clean_df)
    assert len(shift_metrics) == 1
    assert shift_metrics[0].is_severely_distorted
    assert any(r.check_name == "distribution_preservation_audit" and r.status == CheckStatus.WARN for r in check_results)


def test_domain_validator():
    df = pd.DataFrame({
        "age": [25, 30, -5],  # Negative age violates non-negative rule
        "probability": [0.1, 0.5, 1.5],  # > 1.0 violates custom rule
    })

    results = DomainValidator.validate(
        df,
        custom_domain_rules={"probability": (0.0, 1.0)},
    )
    assert any(r.status == CheckStatus.FAIL and "probability" in r.check_name for r in results)
    assert any(r.status == CheckStatus.WARN and "age" in r.check_name for r in results)


def test_data_validation_agent_quality_gate_passed():
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
        "feat_b": [10.0, 20.0, 30.0, 40.0, 50.0] * 20,
        "target": [0, 1] * 50,
    })

    agent = DataValidationAgent()
    gate_report = agent.evaluate_quality_gate(
        raw_df=df,
        cleaned_df=df,
        target_column="target",
    )

    assert gate_report.verdict == QualityGateVerdict.PASSED
    assert gate_report.checks_failed == 0
    assert len(gate_report.blocking_reasons) == 0


def test_data_validation_agent_quality_gate_rejected():
    raw_df = pd.DataFrame({"feat_a": [1.0, 2.0], "target": [0, 1]})
    # Cleaned dataset with missing target and residual NaN
    corrupted_df = pd.DataFrame({"feat_a": [1.0, np.nan]})

    agent = DataValidationAgent()
    gate_report = agent.evaluate_quality_gate(
        raw_df=raw_df,
        cleaned_df=corrupted_df,
        target_column="target",
    )

    assert gate_report.verdict == QualityGateVerdict.REJECTED_BLOCKING
    assert gate_report.checks_failed >= 2
    assert len(gate_report.blocking_reasons) >= 2
