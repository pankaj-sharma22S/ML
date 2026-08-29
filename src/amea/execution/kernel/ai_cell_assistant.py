"""AI assistant for interactive code cell generation and execution result interpretation."""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from amea.execution.kernel.execution_result import CellExecutionResult, CellOutputType
from amea.execution.security import AstSecurityValidator


class AICellSuggestion(BaseModel):
    """AI-generated code cell suggestion with security validation metadata."""
    prompt: str
    code: str
    explanation: str
    is_safe: bool = True
    security_violations: List[str] = Field(default_factory=list)


class AIInterpretation(BaseModel):
    """Factual, evidence-backed interpretation of cell execution outputs."""
    summary: str
    key_findings: List[str] = Field(default_factory=list)
    recommendation: Optional[str] = None


class AICellAssistant:
    """Generates Python code cells for interactive exploration and interprets results."""

    @classmethod
    def generate_cell(
        cls,
        user_prompt: str,
        active_variables: Optional[List[str]] = None,
    ) -> AICellSuggestion:
        """Generate an exploratory or analytical code cell based on natural query."""
        p_lower = user_prompt.lower()
        code = ""
        explanation = ""

        # Pattern-based smart template selection
        if "missing" in p_lower or "null" in p_lower:
            code = "df.isnull().sum()"
            explanation = "Calculates total missing / null values per column in active DataFrame."

        elif "shape" in p_lower or "dimension" in p_lower or "size" in p_lower:
            code = "print(f'Rows: {df.shape[0]}, Columns: {df.shape[1]}')\ndf.dtypes"
            explanation = "Inspects row and column counts and datatypes."

        elif "head" in p_lower or "sample" in p_lower or "preview" in p_lower:
            code = "df.head(10)"
            explanation = "Displays first 10 rows of the DataFrame as an interactive table."

        elif "describe" in p_lower or "summary" in p_lower or "stats" in p_lower:
            code = "df.describe(include='all')"
            explanation = "Generates statistical summary of numerical and categorical variables."

        elif "hist" in p_lower or "distribution" in p_lower or "plot" in p_lower:
            code = """import matplotlib.pyplot as plt
import seaborn as sns

num_cols = df.select_dtypes(include=['number']).columns
if len(num_cols) > 0:
    df[num_cols[:3]].hist(figsize=(10, 4), bins=20)
    plt.tight_layout()
    plt.show()
"""
            explanation = "Plots distribution histograms for numeric variables."

        elif "corr" in p_lower or "relationship" in p_lower:
            code = """import matplotlib.pyplot as plt
import seaborn as sns

numeric_df = df.select_dtypes(include=['number'])
if not numeric_df.empty:
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.show()
"""
            explanation = "Computes Pearson correlation matrix and plots a heatmap."

        elif "imbalance" in p_lower or "class" in p_lower or "target" in p_lower:
            code = """target_col = [c for c in df.columns if 'target' in c.lower() or 'churn' in c.lower() or 'label' in c.lower()]
if target_col:
    print(df[target_col[0]].value_counts(normalize=True) * 100)
else:
    print('Please specify target column: df[\"target\"].value_counts()')
"""
            explanation = "Checks target class distribution and class balance percentage."

        else:
            code = f"# AI exploration cell for: {user_prompt}\ndf.info()"
            explanation = "Inspects column information and non-null counts."

        # Validate security before showing
        violations = AstSecurityValidator.validate_code_safety(code)
        is_safe = (len(violations) == 0)

        return AICellSuggestion(
            prompt=user_prompt,
            code=code.strip(),
            explanation=explanation,
            is_safe=is_safe,
            security_violations=violations,
        )

    @classmethod
    def interpret_result(cls, result: CellExecutionResult) -> AIInterpretation:
        """Provide factual interpretation of raw execution outputs without hiding raw data."""
        if not result.is_success:
            err = result.failure_diagnosis
            return AIInterpretation(
                summary=f"Cell execution failed: {err.root_cause if err else 'Runtime error'}",
                key_findings=["Encountered exception during execution."],
                recommendation=err.recovery_hint if err else "Check variable definitions and script syntax.",
            )

        findings: List[List[str]] = []
        for out in result.outputs:
            if out.output_type == CellOutputType.STREAM and out.text:
                if "missing" in out.text.lower() or "0" in out.text:
                    findings.append("Captured stdout metric or data overview.")
            elif out.output_type == CellOutputType.DATAFRAME and out.dataframe:
                findings.append(f"Loaded DataFrame with {out.dataframe.total_columns} columns and {out.dataframe.total_rows} preview rows.")
            elif out.output_type == CellOutputType.IMAGE:
                findings.append("Generated visualization plot artifact.")
            elif out.output_type == CellOutputType.SCALAR:
                findings.append(f"Calculated scalar metric: {out.scalar_value}")

        flattened = [f for f in findings] or ["Cell executed successfully."]
        return AIInterpretation(
            summary="Execution completed successfully with verifiable output evidence.",
            key_findings=flattened,
            recommendation="Proceed with next exploration step or feature engineering.",
        )
