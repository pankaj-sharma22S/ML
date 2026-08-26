"""Query-First Data Analysis package."""

from amea.query_analysis.schemas import (
    QueryAnalysisRequest,
    QueryAnalysisResponse,
    QueryIntent,
    DatasetProfile,
    DataQualityIssue,
    CleaningAction,
    InsightItem,
    PatternItem,
    RelationshipItem,
    VisualizationArtifact,
)
from amea.query_analysis.query_analyzer import QueryAnalyzer
from amea.query_analysis.ingestion import MultiFileIngestionEngine
from amea.query_analysis.profiler import DataProfilerEngine
from amea.query_analysis.cleaner import EvidenceBasedCleaner
from amea.query_analysis.insights import InsightGenerator
from amea.query_analysis.patterns import PatternDetector
from amea.query_analysis.relationships import RelationshipAnalyzer
from amea.query_analysis.visualization import QueryDrivenVisualizer
from amea.query_analysis.artifact_manager import AnalysisArtifactManager
from amea.query_analysis.service import QueryAnalysisService
from amea.query_analysis.router import router

__all__ = [
    "QueryAnalysisRequest",
    "QueryAnalysisResponse",
    "QueryIntent",
    "DatasetProfile",
    "DataQualityIssue",
    "CleaningAction",
    "InsightItem",
    "PatternItem",
    "RelationshipItem",
    "VisualizationArtifact",
    "QueryAnalyzer",
    "MultiFileIngestionEngine",
    "DataProfilerEngine",
    "EvidenceBasedCleaner",
    "InsightGenerator",
    "PatternDetector",
    "RelationshipAnalyzer",
    "QueryDrivenVisualizer",
    "AnalysisArtifactManager",
    "QueryAnalysisService",
    "router",
]
