"""Service coordinator executing the Query-First Analytical pipeline."""

from typing import Dict, List, Optional
from uuid import uuid4
import pandas as pd

from amea.query_analysis.artifact_manager import AnalysisArtifactManager
from amea.query_analysis.cleaner import EvidenceBasedCleaner
from amea.query_analysis.ingestion import MultiFileIngestionEngine
from amea.query_analysis.insights import InsightGenerator
from amea.query_analysis.patterns import PatternDetector
from amea.query_analysis.profiler import DataProfilerEngine
from amea.query_analysis.query_analyzer import QueryAnalyzer
from amea.query_analysis.relationships import RelationshipAnalyzer
from amea.query_analysis.schemas import (
    CleaningAction,
    DataQualityIssue,
    DatasetProfile,
    InsightItem,
    PatternItem,
    QueryAnalysisRequest,
    QueryAnalysisResponse,
    QueryIntent,
    RelationshipItem,
    VisualizationArtifact,
)
from amea.query_analysis.visualization import QueryDrivenVisualizer


class QueryAnalysisService:
    """End-to-end Query-First Data Analysis pipeline executor."""

    def __init__(self, artifact_manager: Optional[AnalysisArtifactManager] = None):
        self.artifact_manager = artifact_manager or AnalysisArtifactManager()
        self.visualizer = QueryDrivenVisualizer(self.artifact_manager)

    def analyze(self, request: QueryAnalysisRequest) -> QueryAnalysisResponse:
        run_id = f"run_{uuid4().hex[:8]}"

        if not request.file_paths:
            return QueryAnalysisResponse(
                run_id=run_id,
                query=request.query,
                query_intent=QueryIntent(),
                warnings=["No data files provided for analysis."],
                limitations=["Cannot perform analysis without at least one data file."],
            )

        # 1. Ingestion
        ingestion_records = MultiFileIngestionEngine.ingest_files(request.file_paths)

        # 2. Data Profiling
        dataset_profiles: List[DatasetProfile] = [
            DataProfilerEngine.profile_dataset(rec) for rec in ingestion_records
        ]

        all_cols: List[str] = []
        for prof in dataset_profiles:
            all_cols.extend(prof.column_names)

        # 3. Query Understanding
        intent = QueryAnalyzer.analyze(request.query, available_columns=all_cols)

        # 4. Ambiguity check & human clarification escalation
        if intent.is_ambiguous:
            return QueryAnalysisResponse(
                run_id=run_id,
                query=request.query,
                query_intent=intent,
                datasets=dataset_profiles,
                clarification_required=True,
                clarification_question=intent.clarification_question,
                warnings=["Query is ambiguous with multiple candidate columns."],
            )

        # 5. Data Quality Audit & Evidence-Based Cleaning
        all_issues: List[DataQualityIssue] = []
        cleaned_dfs: Dict[str, pd.DataFrame] = {}
        all_cleaning_actions: List[CleaningAction] = []

        for rec, prof in zip(ingestion_records, dataset_profiles):
            issues = EvidenceBasedCleaner.audit_quality(rec, prof, intent)
            all_issues.extend(issues)
            cleaned_df, actions = EvidenceBasedCleaner.clean_dataset(rec, issues)
            cleaned_dfs[rec.dataset_id] = cleaned_df
            all_cleaning_actions.extend(actions)

        # 6. Concrete Statistical Insights
        insights: List[InsightItem] = InsightGenerator.generate_insights(
            dfs=cleaned_dfs,
            profiles=dataset_profiles,
            intent=intent,
        )

        # 7. Pattern Detection
        patterns: List[PatternItem] = PatternDetector.detect_patterns(
            dfs=cleaned_dfs,
            profiles=dataset_profiles,
            intent=intent,
        )

        # 8. Relationship Analysis
        relationships: List[RelationshipItem] = RelationshipAnalyzer.analyze_relationships(
            dfs=cleaned_dfs,
            profiles=dataset_profiles,
        )

        # 9. Query-Driven Selective Visualizations
        visualizations: List[VisualizationArtifact] = self.visualizer.generate_visualizations(
            dfs=cleaned_dfs,
            profiles=dataset_profiles,
            intent=intent,
            run_id=run_id,
        )

        return QueryAnalysisResponse(
            run_id=run_id,
            query=request.query,
            query_intent=intent,
            datasets=dataset_profiles,
            data_quality=all_issues,
            cleaning_actions=all_cleaning_actions,
            relationships=relationships,
            insights=insights,
            patterns=patterns,
            visualizations=visualizations,
        )
