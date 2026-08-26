"""Code Synthesis Agent coordinating pipeline code generation and artifact persistence."""

from pathlib import Path
from typing import Optional
from uuid import uuid4

from amea.code_synthesis.models import (
    CodeSynthesisContext,
    GeneratedCodeArtifacts,
)
from amea.code_synthesis.templates import PipelineTemplateEngine
from amea.code_synthesis.validator import CodeSyntaxValidator


class CodeSynthesisAgent:
    """Independent agent transforming validated upstream ML decisions into production code."""

    def __init__(self, base_output_dir: Optional[Path] = None):
        self.base_output_dir = (base_output_dir or Path(".amea_project/generated")).resolve()
        self.template_engine = PipelineTemplateEngine()
        self.validator = CodeSyntaxValidator()

    def synthesize(self, context: CodeSynthesisContext) -> GeneratedCodeArtifacts:
        """Synthesize, validate, and persist the complete standalone production pipeline."""
        pipeline_id = f"pipeline_{uuid4().hex[:8]}"
        pipeline_dir = self.base_output_dir / pipeline_id
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        # 1. Render all modular pipeline files
        files = {
            "data_loader.py": self.template_engine.render_data_loader(context),
            "preprocess.py": self.template_engine.render_preprocess(context),
            "features.py": self.template_engine.render_features(context),
            "train.py": self.template_engine.render_train(context),
            "evaluate.py": self.template_engine.render_evaluate(context),
            "inference.py": self.template_engine.render_inference(context),
            "requirements.txt": self.template_engine.render_requirements(context),
            "config.json": self.template_engine.render_config(context, pipeline_id),
        }

        # 2. Validate AST and semantic alignment
        validation_report = self.validator.validate_pipeline_code(files, context)

        # 3. Persist files to disk
        for filename, content in files.items():
            file_path = pipeline_dir / filename
            file_path.write_text(content, encoding="utf-8")

        return GeneratedCodeArtifacts(
            pipeline_id=pipeline_id,
            files=files,
            pipeline_dir=str(pipeline_dir.resolve()),
            metadata={
                "target_column": context.task_spec.target_column,
                "task_type": context.task_spec.task_type.value,
                "model_family": context.best_candidate.model_family,
                "best_cv_score": context.best_candidate.cv_metrics_mean.get(context.task_spec.primary_metric, 0.0),
            },
            validation_report=validation_report,
        )
