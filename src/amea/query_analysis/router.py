"""API endpoint router and request handler for Query-First Data Analysis."""

from typing import Any, Dict
from amea.query_analysis.schemas import QueryAnalysisRequest, QueryAnalysisResponse
from amea.query_analysis.service import QueryAnalysisService


class QueryAnalysisRouter:
    """REST API dispatcher for Query-First Data Analysis."""

    def __init__(self, prefix: str = "/api/query-analysis"):
        self.prefix = prefix
        self.service = QueryAnalysisService()

    def analyze(self, request: QueryAnalysisRequest) -> QueryAnalysisResponse:
        """
        POST /api/query-analysis/analyze
        Executes query-driven multi-file analytical pipeline.
        """
        return self.service.analyze(request)


# Global singleton instance for endpoint routing
router = QueryAnalysisRouter()


def analyze_query_data(request: QueryAnalysisRequest) -> QueryAnalysisResponse:
    """Public functional endpoint handler."""
    return router.analyze(request)
