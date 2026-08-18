"""Natural language user intent parser."""

import re
from typing import List, Optional
from amea.problem_understanding.models import IntentAnalysis, IntentCategory


class IntentParser:
    """Extracts structured intent, metrics, and candidate targets from user prompts."""

    @staticmethod
    def parse(user_query: str) -> IntentAnalysis:
        query_lower = user_query.lower()

        # 1. Intent Category Extraction
        if any(w in query_lower for w in ["forecast", "time series", "future demand", "temporal"]):
            category = IntentCategory.FORECASTING
        elif any(w in query_lower for w in ["regress", "continuous", "price", "revenue", "salary", "cost", "score", "estimate"]):
            category = IntentCategory.REGRESSION
        elif any(w in query_lower for w in ["classify", "classification", "churn", "fraud", "binary", "multiclass", "category"]):
            category = IntentCategory.CLASSIFICATION
        elif any(w in query_lower for w in ["cluster", "grouping", "segment"]):
            category = IntentCategory.CLUSTERING
        elif any(w in query_lower for w in ["anomaly", "outlier detection"]):
            category = IntentCategory.ANOMALY_DETECTION
        else:
            category = IntentCategory.PREDICTION

        # 2. Extract potential target column name mentioned in prompt
        target_candidate = None
        
        # High confidence target keywords
        known_target_keywords = ["price", "churn", "fraud", "salary", "revenue", "cost", "sales", "default", "rating"]
        for kw in known_target_keywords:
            if re.search(r"\b" + kw + r"\b", query_lower):
                target_candidate = kw
                break

        if not target_candidate:
            target_patterns = [
                r"(?:predict|target|classify|forecast|detect|estimate)\s+(?:the\s+)?([a-zA-Z0-9_\-]+)",
                r"target(?:_column)?\s*[:=]\s*([a-zA-Z0-9_\-]+)",
                r"for\s+([a-zA-Z0-9_\-]+)\s+prediction",
            ]
            for pattern in target_patterns:
                match = re.search(pattern, user_query, re.IGNORECASE)
                if match:
                    cand = match.group(1).strip()
                    if cand.lower() not in ["a", "an", "the", "model", "pipeline", "dataset", "data", "binary", "multiclass", "baseline", "continuous"]:
                        target_candidate = cand
                        break

        # 3. Extract requested metrics (only explicit phrases like "metric: accuracy" or "optimize roc_auc")
        metrics = []
        metric_keywords = ["roc_auc", "f1_macro", "logloss", "rmse", "mae", "r2", "pr_auc"]
        for m in metric_keywords:
            if m in query_lower:
                metrics.append(m)

        if "accuracy" in query_lower and ("metric" in query_lower or "optimize" in query_lower):
            metrics.append("accuracy")

        # 4. Domain keywords
        domain_keywords = []
        domain_terms = ["churn", "fraud", "credit", "sales", "medical", "finance", "telecom", "retail", "iot", "real estate"]
        for d in domain_terms:
            if d in query_lower:
                domain_keywords.append(d)

        return IntentAnalysis(
            raw_query=user_query,
            primary_intent=category,
            mentioned_target_candidate=target_candidate,
            requested_metrics=metrics,
            stated_constraints=[],
            domain_keywords=domain_keywords,
        )
