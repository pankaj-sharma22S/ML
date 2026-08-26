"""Comprehensive unit and integration tests for Query-First Data Analysis."""

from pathlib import Path
import pytest

from amea.query_analysis.ingestion import MultiFileIngestionEngine
from amea.query_analysis.query_analyzer import QueryAnalyzer
from amea.query_analysis.profiler import DataProfilerEngine
from amea.query_analysis.cleaner import EvidenceBasedCleaner
from amea.query_analysis.insights import InsightGenerator
from amea.query_analysis.patterns import PatternDetector
from amea.query_analysis.relationships import RelationshipAnalyzer
from amea.query_analysis.service import QueryAnalysisService
from amea.query_analysis.schemas import QueryAnalysisRequest
from amea.query_analysis.router import router, analyze_query_data


@pytest.fixture
def sample_sales_csv(tmp_path):
    p = tmp_path / "sales.csv"
    content = """date,product,revenue,units,region
2024-01-01,ProductA,10000,100,North
2024-02-01,ProductA,12000,120,North
2024-03-01,ProductA,8000,80,North
2024-01-01,ProductB,5000,50,South
2024-02-01,ProductB,6000,60,South
2024-03-01,ProductB,4000,40,South
2024-03-01,ProductB,4000,40,South
"""  # Contains a duplicate row
    p.write_text(content.strip(), encoding="utf-8")
    return str(p)


@pytest.fixture
def sample_products_csv(tmp_path):
    p = tmp_path / "products.csv"
    content = """product,category,cost_price
ProductA,Electronics,50
ProductB,Apparel,20
ProductC,HomeGoods,15
"""
    p.write_text(content.strip(), encoding="utf-8")
    return str(p)


def test_query_analyzer_intent_and_metrics():
    # Trend query
    q1 = "Analyze revenue trend over time"
    intent1 = QueryAnalyzer.analyze(q1)
    assert intent1.primary_intent == "trend_analysis"
    assert "revenue" in intent1.target_metrics

    # Ranking query
    q2 = "Which products caused the revenue decline?"
    intent2 = QueryAnalyzer.analyze(q2)
    assert intent2.primary_intent == "ranking" or "ranking" in intent2.secondary_intents
    assert "product" in intent2.target_dimensions

    # Correlation query
    q3 = "Show correlation between revenue and units"
    intent3 = QueryAnalyzer.analyze(q3)
    assert intent3.primary_intent == "correlation"


def test_query_analyzer_ambiguity_detection():
    query = "Analyze revenue breakdown"
    available_cols = ["gross_revenue", "net_revenue", "product"]
    intent = QueryAnalyzer.analyze(query, available_columns=available_cols)
    assert intent.is_ambiguous is True
    assert intent.clarification_question is not None


def test_ingestion_and_profiler(sample_sales_csv):
    record = MultiFileIngestionEngine.ingest_file(sample_sales_csv)
    assert record.file_type == "csv"
    assert len(record.df) == 7

    profile = DataProfilerEngine.profile_dataset(record)
    assert profile.rows == 7
    assert profile.columns == 5
    assert "revenue" in profile.numeric_columns
    assert "date" in profile.datetime_columns
    assert profile.duplicate_rows_count == 1


def test_evidence_based_cleaning(sample_sales_csv):
    record = MultiFileIngestionEngine.ingest_file(sample_sales_csv)
    profile = DataProfilerEngine.profile_dataset(record)
    intent = QueryAnalyzer.analyze("Analyze total revenue by product")

    issues = EvidenceBasedCleaner.audit_quality(record, profile, intent)
    assert len(issues) > 0
    assert any(i.issue_type == "duplicate_rows" and i.impacts_user_query for i in issues)

    cleaned_df, actions = EvidenceBasedCleaner.clean_dataset(record, issues)
    assert len(cleaned_df) == 6  # Duplicate dropped
    assert len(actions) == 1
    assert actions[0].operation == "drop_duplicates"


def test_insight_and_pattern_generation(sample_sales_csv):
    record = MultiFileIngestionEngine.ingest_file(sample_sales_csv)
    profile = DataProfilerEngine.profile_dataset(record)
    intent = QueryAnalyzer.analyze("Which product drives revenue?")

    cleaned_df, _ = EvidenceBasedCleaner.clean_dataset(record, [])
    dfs = {record.dataset_id: cleaned_df}

    insights = InsightGenerator.generate_insights(dfs, [profile], intent)
    assert len(insights) > 0
    assert "ProductA" in insights[0].insight
    assert insights[0].calculation["share_percentage"] > 0

    patterns = PatternDetector.detect_patterns(dfs, [profile], intent)
    assert isinstance(patterns, list)


def test_relationship_analyzer(sample_sales_csv, sample_products_csv):
    rec_sales = MultiFileIngestionEngine.ingest_file(sample_sales_csv)
    rec_prod = MultiFileIngestionEngine.ingest_file(sample_products_csv)

    prof_sales = DataProfilerEngine.profile_dataset(rec_sales)
    prof_prod = DataProfilerEngine.profile_dataset(rec_prod)

    dfs = {rec_sales.dataset_id: rec_sales.df, rec_prod.dataset_id: rec_prod.df}
    relationships = RelationshipAnalyzer.analyze_relationships(dfs, [prof_sales, prof_prod])

    assert len(relationships) > 0
    # Should detect shared key 'product' between sales.csv and products.csv
    shared_rel = next((r for r in relationships if r.relationship_type == "shared_identifier"), None)
    assert shared_rel is not None
    assert shared_rel.column_a == "product"


def test_query_analysis_service_end_to_end(sample_sales_csv, sample_products_csv, tmp_path):
    service = QueryAnalysisService()
    req = QueryAnalysisRequest(
        query="Analyze revenue trend and show product ranking",
        file_paths=[sample_sales_csv, sample_products_csv],
    )

    response = service.analyze(req)
    assert response.run_id is not None
    assert len(response.datasets) == 2
    assert len(response.insights) > 0
    assert len(response.visualizations) > 0
    assert any(v.chart_type in ["line_chart", "bar_chart"] for v in response.visualizations)


def test_router_endpoint_execution(sample_sales_csv):
    payload = QueryAnalysisRequest(
        query="Show correlation between revenue and units",
        file_paths=[sample_sales_csv],
    )

    response = router.analyze(payload)
    assert response.query_intent.primary_intent == "correlation"
    assert len(response.datasets) == 1
    assert len(response.visualizations) > 0

    # Test direct function entrypoint
    func_resp = analyze_query_data(payload)
    assert func_resp.run_id is not None
