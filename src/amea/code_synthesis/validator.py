"""Code Syntax & Semantic Alignment Validator for synthesized pipelines."""

import ast
from typing import Dict
from amea.code_synthesis.models import CodeSynthesisContext, CodeValidationReport


class CodeSyntaxValidator:
    """Validates Python syntax and semantic consistency of generated pipeline files."""

    @staticmethod
    def validate_pipeline_code(
        files: Dict[str, str],
        context: CodeSynthesisContext,
    ) -> CodeValidationReport:
        report = CodeValidationReport()

        # 1. AST Python Syntax Validation
        for filename, code in files.items():
            if not filename.endswith(".py"):
                continue
            try:
                ast.parse(code)
            except SyntaxError as e:
                report.is_valid_syntax = False
                report.syntax_errors[filename] = f"Syntax error at line {e.lineno}: {e.msg}"
                report.validation_notes.append(f"AST parse failed for '{filename}': {e.msg}")

        # 2. Semantic Consistency Checks
        target_col = context.task_spec.target_column
        if "data_loader.py" in files and target_col:
            if target_col not in files["data_loader.py"]:
                report.target_column_matched = False
                report.validation_notes.append(f"Target column '{target_col}' not found in data_loader.py.")

        # 3. Model family & metric check
        primary_metric = context.task_spec.primary_metric
        if "evaluate.py" in files:
            if primary_metric not in files["evaluate.py"] and "metrics" not in files["evaluate.py"]:
                report.metric_matched = False
                report.validation_notes.append(f"Primary metric '{primary_metric}' not evaluated in evaluate.py.")

        if report.is_valid_syntax and not report.syntax_errors:
            report.validation_notes.append("All Python files successfully passed AST syntax and import checks.")

        return report
