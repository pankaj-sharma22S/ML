"""Data Validation & Quality Gate Agent executing independent pre-modeling quality audits."""

from typing import Dict, List, Optional, Set
import pandas as pd

from amea.data_validation.models import (
    CheckStatus,
    QualityGateReport,
    QualityGateVerdict,
    ValidationCheckResult,
)
from amea.data_validation.schema_validator import SchemaValidator
from amea.data_validation.integrity_validator import IntegrityValidator
from amea.data_validation.distribution_auditor import DistributionAuditor
from amea.data_validation.domain_validator import DomainValidator


class DataValidationAgent:
    """Independent Quality Gate determining if a cleaned dataset is certified for ML modeling."""

    def __init__(self):
        self.schema_validator = SchemaValidator()
        self.integrity_validator = IntegrityValidator()
        self.distribution_auditor = DistributionAuditor()
        self.domain_validator = DomainValidator()

    def evaluate_quality_gate(
        self,
        raw_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        dataset_name: str = "dataset",
        target_column: Optional[str] = None,
        expected_dropped_columns: Optional[Set[str]] = None,
        custom_domain_rules: Optional[Dict[str, tuple[Optional[float], Optional[float]]]] = None,
    ) -> QualityGateReport:
        """Run all validation suites and produce authoritative QualityGateReport."""
        all_checks: List[ValidationCheckResult] = []

        # 1. Schema Validation
        schema_checks = self.schema_validator.validate(
            raw_df=raw_df,
            cleaned_df=cleaned_df,
            target_column=target_column,
            expected_dropped_columns=expected_dropped_columns,
        )
        all_checks.extend(schema_checks)

        # 2. Row & Missingness Integrity Validation
        integrity_checks = self.integrity_validator.validate(
            raw_df=raw_df,
            cleaned_df=cleaned_df,
            target_column=target_column,
        )
        all_checks.extend(integrity_checks)

        # 3. Distribution Preservation Audit
        shift_metrics, dist_checks = self.distribution_auditor.audit(
            raw_df=raw_df,
            cleaned_df=cleaned_df,
        )
        all_checks.extend(dist_checks)

        # 4. Domain & Boundary Constraint Validation
        domain_checks = self.domain_validator.validate(
            df=cleaned_df,
            custom_domain_rules=custom_domain_rules,
        )
        all_checks.extend(domain_checks)

        # Compile statistics & determine verdict
        passed_count = sum(1 for c in all_checks if c.status == CheckStatus.PASS)
        warn_count = sum(1 for c in all_checks if c.status == CheckStatus.WARN)
        fail_count = sum(1 for c in all_checks if c.status == CheckStatus.FAIL)
        blocking_checks = [c for c in all_checks if c.is_blocking and c.status == CheckStatus.FAIL]

        blocking_reasons = [f"[{c.check_name}] {c.message}" for c in blocking_checks]
        recommendations = []

        if blocking_checks:
            verdict = QualityGateVerdict.REJECTED_BLOCKING
            recommendations.append("Quality Gate REJECTED: Dataset cannot safely proceed to ML training. Resolve blocking schema or missingness errors.")
        elif warn_count > 0:
            verdict = QualityGateVerdict.WARNING_PROCEED
            recommendations.append(f"Quality Gate PASSED with {warn_count} warnings. Proceed with caution and log caveats in model audit report.")
        else:
            verdict = QualityGateVerdict.PASSED
            recommendations.append("Quality Gate PASSED cleanly: Dataset is certified ready for ML Strategy and Feature Engineering.")

        return QualityGateReport(
            verdict=verdict,
            dataset_name=dataset_name,
            total_checks_run=len(all_checks),
            checks_passed=passed_count,
            checks_warned=warn_count,
            checks_failed=fail_count,
            check_results=all_checks,
            distribution_shifts=shift_metrics,
            blocking_reasons=blocking_reasons,
            recommendations_for_orchestrator=recommendations,
        )
