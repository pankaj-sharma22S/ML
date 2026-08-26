"""Adaptive Query Complexity Classifier and Execution Planner."""

import re
from typing import List, Tuple


class AdaptiveQueryPlanner:
    """Classifies query complexity and selects execution pathway (Direct vs Agentic Multi-Step)."""

    COMPLEX_INDICATORS = [
        r"\bcompare\b",
        r"\bdifference\b",
        r"\bwhy\s+did\b",
        r"\bdecline\b",
        r"\btrend\b",
        r"\bacross\s+all\b",
        r"\bhow\s+are\s+.*?\s+related\b",
        r"\broot\s+cause\b",
        r"\bconflict\b",
        r"\bdiscrepancy\b",
        r"\bmulti-hop\b",
    ]

    @classmethod
    def analyze_query(cls, query: str) -> Tuple[bool, List[str]]:
        """
        Returns (is_complex, required_capabilities).
        is_complex = False -> Direct Simple Retrieval
        is_complex = True  -> Multi-step Agentic RAG
        """
        lower = query.lower()
        is_complex = False
        reasons = []

        for pattern in cls.COMPLEX_INDICATORS:
            if re.search(pattern, lower):
                is_complex = True
                reasons.append(f"Matched complexity trigger: '{pattern}'")

        # Multi-question detection
        if "?" in query and query.count("?") > 1:
            is_complex = True
            reasons.append("Multi-part question detected.")

        if " and " in lower and ("what" in lower or "how" in lower or "why" in lower):
            is_complex = True
            reasons.append("Compound inquiry detected.")

        return is_complex, reasons
