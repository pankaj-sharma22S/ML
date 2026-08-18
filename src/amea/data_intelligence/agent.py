"""Data Intelligence Agent coordinating end-to-end data auditing and packaging."""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from amea.core.state import DataProfile
from amea.data_intelligence.lineage import DatasetLineageManager
from amea.data_intelligence.profiler import DeepDataProfiler
from amea.data_intelligence.quality_auditor import QualityAuditor
from amea.data_intelligence.leakage_guard import LeakageGuard
from amea.data_intelligence.relationship_miner import RelationshipMiner
from amea.data_intelligence.strategy_recommender import StrategyRecommender
from amea.data_intelligence.models import DataEvidencePackage


class DataIntelligenceAgent:
    """Discovers, profiles, audits, and packages dataset evidence for the Orchestrator."""

    def __init__(self):
        self.lineage_mgr = DatasetLineageManager()
        self.profiler = DeepDataProfiler()
        self.auditor = QualityAuditor()
        self.leakage_guard = LeakageGuard()
        self.relationship_miner = RelationshipMiner()
        self.recommender = StrategyRecommender()

    def process_dataset(
        self,
        dataset_path: Path | str,
        target_column: Optional[str] = None,
        is_classification: bool = True,
    ) -> Tuple[DataEvidencePackage, DataProfile]:
        """Execute complete data intelligence pipeline on a dataset file."""
        path_obj = Path(dataset_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")

        # 1. Read dataset (Immutable read)
        df = pd.read_csv(path_obj)

        # 2. Lineage & Cryptographic Versioning
        version_meta = self.lineage_mgr.create_version(source_path=path_obj, df=df)

        # 3. Deep Deterministic Statistical Profiling
        data_profile = self.profiler.profile(
            df=df,
            dataset_path=str(path_obj.resolve()),
            sha256=version_meta.source_hash_sha256,
        )

        # 4. Quality & Missingness Audit
        quality_audit = self.auditor.audit(df=df)

        # 5. Target Leakage & ID Audit
        leakage_findings = self.leakage_guard.audit_leakage(
            df=df,
            target_col=target_column,
            is_classification=is_classification,
        )

        # 6. Inter-Feature Relationship Mining
        relationships = self.relationship_miner.mine_relationships(df=df)

        # 7. Evidence-Backed Strategy Recommendations
        treatments = self.recommender.recommend(
            quality_audit=quality_audit,
            leakage_findings=leakage_findings,
            relationships=relationships,
        )

        # 8. Identify Blocking Issues
        blocking_issues = []
        if len(df) == 0:
            blocking_issues.append("Dataset is empty (0 rows).")
        if target_column and target_column not in df.columns:
            blocking_issues.append(f"Specified target column '{target_column}' is missing from schema.")

        # 9. Summary Findings
        summary = [
            f"Dataset contains {len(df)} rows and {len(df.columns)} columns (memory: {data_profile.memory_footprint_mb} MB).",
            f"Data Quality Score: {quality_audit.quality_score * 100:.0f}% (Clean: {quality_audit.is_clean}).",
        ]
        if quality_audit.missingness_findings:
            summary.append(f"Missing values diagnosed in {len(quality_audit.missingness_findings)} columns.")
        if leakage_findings:
            summary.append(f"Leakage Guard detected {len(leakage_findings)} potential risk features.")
        if relationships:
            summary.append(f"Identified {len(relationships)} collinear feature pairs.")

        evidence_package = DataEvidencePackage(
            dataset_version=version_meta,
            total_rows=len(df),
            total_columns=len(df.columns),
            memory_mb=data_profile.memory_footprint_mb,
            quality_audit=quality_audit,
            leakage_findings=leakage_findings,
            relationship_findings=relationships,
            treatment_candidates=treatments,
            blocking_issues=blocking_issues,
            summary_findings=summary,
        )

        return evidence_package, data_profile
