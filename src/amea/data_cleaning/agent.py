"""Data Cleaning Agent orchestrating transformation, validation, and versioning."""

from pathlib import Path
from typing import List, Optional
import pandas as pd

from amea.data_cleaning.models import CleanedDataArtifact, CleaningPlan, CleaningAction
from amea.data_cleaning.pipeline_builder import CleaningPipelineBuilder
from amea.data_cleaning.validator import PostCleaningValidator
from amea.data_intelligence.lineage import DatasetLineageManager
from amea.data_intelligence.models import DataTreatmentCandidate, DatasetVersion


class DataCleaningAgent:
    """Execution agent converting approved cleaning strategies into reproducible, validated datasets."""

    def __init__(self, persistence_dir: Optional[Path] = None):
        self.persistence_dir = (persistence_dir or Path(".amea_project")).resolve()
        self.datasets_dir = self.persistence_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.lineage_mgr = DatasetLineageManager()
        self.validator = PostCleaningValidator()

    def clean_dataset(
        self,
        raw_dataset_path: Path | str,
        treatment_candidates: Optional[List[DataTreatmentCandidate]] = None,
        cleaning_plan: Optional[CleaningPlan] = None,
        target_column: Optional[str] = None,
        parent_version: Optional[DatasetVersion] = None,
    ) -> CleanedDataArtifact:
        """Fit cleaning pipeline, transform dataset, validate, and serialize new immutable version."""
        path_obj = Path(raw_dataset_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Raw dataset '{raw_dataset_path}' not found.")

        # 1. Read raw source
        raw_df = pd.read_csv(path_obj)

        # 2. Build Pipeline
        if cleaning_plan:
            pipeline = CleaningPipelineBuilder.build_from_plan(cleaning_plan)
            applied_plan = cleaning_plan
        else:
            candidates = treatment_candidates or []
            pipeline = CleaningPipelineBuilder.build_from_treatments(candidates)
            applied_plan = CleaningPlan(
                plan_id="plan_from_candidates",
                actions=[],
            )

        # 3. Fit & Transform
        # Separate target if present so target values are not accidentally imputed/modified
        if target_column and target_column in raw_df.columns:
            features_df = raw_df.drop(columns=[target_column])
            target_series = raw_df[target_column]
            cleaned_features = pipeline.fit_transform(features_df)
            if not isinstance(cleaned_features, pd.DataFrame):
                cleaned_features = pd.DataFrame(cleaned_features, columns=[c for c in features_df.columns if c in features_df.columns])
            cleaned_df = cleaned_features.copy()
            cleaned_df[target_column] = target_series.values
        else:
            cleaned_df = pipeline.fit_transform(raw_df)
            if not isinstance(cleaned_df, pd.DataFrame):
                cleaned_df = pd.DataFrame(cleaned_df, columns=raw_df.columns)

        # 4. Post-Cleaning Validation Audit
        validation_report = self.validator.validate(
            initial_df=raw_df,
            cleaned_df=cleaned_df,
            target_column=target_column,
        )

        # 5. Cryptographic Versioning & Storage
        parent_id = parent_version.version_id if parent_version else None
        clean_version = self.lineage_mgr.create_version(
            source_path=path_obj,
            df=cleaned_df,
            parent_version_id=parent_id,
            transformation_history=["data_cleaning_pipeline_fit_transform"],
        )

        cleaned_file_path = self.datasets_dir / f"cleaned_{clean_version.version_id}.csv"
        cleaned_df.to_csv(cleaned_file_path, index=False)

        return CleanedDataArtifact(
            cleaned_dataset_path=str(cleaned_file_path.resolve()),
            dataset_version=clean_version,
            validation_report=validation_report,
            applied_cleaning_plan=applied_plan,
        )
