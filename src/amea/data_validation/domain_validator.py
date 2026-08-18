"""Domain and physical boundary constraint validator."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from amea.data_validation.models import CheckStatus, ValidationCheckResult


class DomainValidator:
    """Validates physical domain bounds (e.g. non-negative age/counts/prices) without fabricating rules."""

    @staticmethod
    def validate(
        df: pd.DataFrame,
        custom_domain_rules: Optional[Dict[str, tuple[Optional[float], Optional[float]]]] = None,
    ) -> List[ValidationCheckResult]:
        results: List[ValidationCheckResult] = []
        rules = custom_domain_rules or {}

        # Heuristic non-negative keywords
        non_negative_keywords = ["age", "count", "price", "income", "salary", "distance", "duration", "quantity"]

        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            vals = df[col].dropna().values
            if len(vals) == 0:
                continue

            # 1. Custom configured rules
            if col in rules:
                min_b, max_b = rules[col]
                if min_b is not None and np.any(vals < min_b):
                    violation_count = int(np.sum(vals < min_b))
                    results.append(
                        ValidationCheckResult(
                            check_name=f"domain_rule_{col}",
                            category="domain",
                            status=CheckStatus.FAIL,
                            message=f"CRITICAL: Column '{col}' violates lower domain bound ({min_b}) with {violation_count} values.",
                            evidence={"column": col, "min_bound": min_b, "violations": violation_count},
                            is_blocking=True,
                        )
                    )
                elif max_b is not None and np.any(vals > max_b):
                    violation_count = int(np.sum(vals > max_b))
                    results.append(
                        ValidationCheckResult(
                            check_name=f"domain_rule_{col}",
                            category="domain",
                            status=CheckStatus.FAIL,
                            message=f"CRITICAL: Column '{col}' violates upper domain bound ({max_b}) with {violation_count} values.",
                            evidence={"column": col, "max_bound": max_b, "violations": violation_count},
                            is_blocking=True,
                        )
                    )

            # 2. Standard heuristic non-negative check for standard keywords
            elif any(kw in col.lower() for kw in non_negative_keywords):
                if np.any(vals < 0):
                    neg_count = int(np.sum(vals < 0))
                    results.append(
                        ValidationCheckResult(
                            check_name=f"heuristic_domain_bounds_{col}",
                            category="domain",
                            status=CheckStatus.WARN,
                            message=f"Warning: Semantic domain keyword in '{col}' expects non-negative values, but {neg_count} negative entries found.",
                            evidence={"column": col, "negative_count": neg_count},
                            is_blocking=False,
                        )
                    )

        if not results:
            results.append(
                ValidationCheckResult(
                    check_name="domain_boundary_checks",
                    category="domain",
                    status=CheckStatus.PASS,
                    message="All domain constraints and heuristic value boundaries validated cleanly.",
                    evidence={},
                    is_blocking=False,
                )
            )

        return results
