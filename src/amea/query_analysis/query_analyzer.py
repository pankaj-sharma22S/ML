"""Natural language query understanding and intent extraction."""

import re
from typing import List, Optional, Set
from amea.query_analysis.schemas import QueryIntent


class QueryAnalyzer:
    """Extracts analytical intent, target metrics, dimensions, and ambiguities from a natural query."""

    INTENT_KEYWORDS = {
        "trend_analysis": [r"\btrend\b", r"\bover time\b", r"\bdecline\b", r"\bgrowth\b", r"\bmonthly\b", r"\byearly\b", r"\bquarterly\b", r"\bhistory\b"],
        "ranking": [r"\brank\b", r"\btop\b", r"\bwhich\s+\w+\b", r"\bcaused\b", r"\bdrivers?\b", r"\bhighest\b", r"\blowest\b", r"\bleading\b"],
        "correlation": [r"\bcorrelation\b", r"\brelationship\b", r"\brelated\b", r"\bimpact\b", r"\bassociation\b", r"\bdependencies\b"],
        "distribution": [r"\bdistribution\b", r"\bspread\b", r"\brange\b", r"\bvariance\b", r"\bhistogram\b"],
        "comparison": [r"\bcompare\b", r"\bcomparison\b", r"\bversus\b", r"\bvs\b", r"\bdifference\b"],
        "anomaly_detection": [r"\bunusual\b", r"\boutlier\b", r"\banomal(y|ies)\b", r"\birregular\b", r"\bspike\b", r"\bdip\b"],
        "aggregation": [r"\btotal\b", r"\bsum\b", r"\baverage\b", r"\bmean\b", r"\bcount\b", r"\bbreakdown\b"],
    }

    COMMON_METRICS = [
        "revenue", "sales", "profit", "cost", "margin", "churn", "price", "units",
        "quantity", "amount", "score", "volume", "discount", "loss", "gdp", "balance"
    ]

    COMMON_DIMENSIONS = [
        "product", "region", "country", "date", "month", "year", "quarter", "category",
        "segment", "customer", "department", "channel", "brand", "state", "city"
    ]

    @classmethod
    def analyze(cls, query: str, available_columns: Optional[List[str]] = None) -> QueryIntent:
        """Parse query string into structured QueryIntent."""
        q_lower = query.lower()
        matched_intents: List[str] = []

        for intent, patterns in cls.INTENT_KEYWORDS.items():
            for p in patterns:
                if re.search(p, q_lower):
                    if intent not in matched_intents:
                        matched_intents.append(intent)
                    break

        primary = matched_intents[0] if matched_intents else "aggregation"
        secondary = matched_intents[1:] if len(matched_intents) > 1 else []

        # Target metrics detection
        metrics: List[str] = []
        for m in cls.COMMON_METRICS:
            if re.search(rf"\b{m}\w*\b", q_lower):
                metrics.append(m)

        # Target dimensions detection
        dims: List[str] = []
        for d in cls.COMMON_DIMENSIONS:
            if re.search(rf"\b{d}\w*\b", q_lower):
                dims.append(d)

        # Match against actual dataset columns if provided
        if available_columns:
            for col in available_columns:
                c_lower = col.lower()
                if c_lower in q_lower:
                    if any(m in c_lower for m in cls.COMMON_METRICS):
                        if col not in metrics:
                            metrics.append(col)
                    else:
                        if col not in dims:
                            dims.append(col)

        # Ambiguity check
        is_ambiguous = False
        clarification_q = None

        if "revenue" in q_lower and available_columns:
            rev_matches = [c for c in available_columns if "revenue" in c.lower()]
            if len(rev_matches) > 1 and not any(r.lower() in q_lower for r in rev_matches):
                is_ambiguous = True
                clarification_q = f"Multiple revenue fields found: {rev_matches}. Which one should be analyzed?"

        return QueryIntent(
            primary_intent=primary,
            secondary_intents=secondary,
            target_metrics=metrics,
            target_dimensions=dims,
            is_ambiguous=is_ambiguous,
            clarification_question=clarification_q,
            confidence=0.90 if matched_intents else 0.60,
        )
