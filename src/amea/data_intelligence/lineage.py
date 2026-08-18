"""Dataset lineage and cryptographic versioning."""

import hashlib
import uuid
from pathlib import Path
from typing import List, Optional
import pandas as pd

from amea.data_intelligence.models import DatasetVersion


class DatasetLineageManager:
    """Manages immutable dataset hashing, versioning, and provenance."""

    @staticmethod
    def create_version(
        source_path: Path | str,
        df: pd.DataFrame,
        parent_version_id: Optional[str] = None,
        transformation_history: Optional[List[str]] = None,
    ) -> DatasetVersion:
        """Construct an immutable DatasetVersion record from DataFrame and source path."""
        path_obj = Path(source_path)

        # 1. Content Hash (SHA-256)
        if path_obj.exists() and path_obj.is_file():
            content_hash = hashlib.sha256(path_obj.read_bytes()).hexdigest()
        else:
            # Hash string representation of dataframe values
            content_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

        # 2. Schema Hash (Column names and data types)
        schema_repr = "|".join(f"{col}:{df[col].dtype}" for col in df.columns)
        schema_hash = hashlib.sha256(schema_repr.encode("utf-8")).hexdigest()

        version_id = f"v_{content_hash[:12]}"
        dataset_id = path_obj.stem if path_obj.exists() else f"dataset_{uuid.uuid4().hex[:8]}"

        return DatasetVersion(
            dataset_id=dataset_id,
            version_id=version_id,
            source_uri=str(path_obj.resolve()) if path_obj.exists() else "memory://dataframe",
            source_hash_sha256=content_hash,
            schema_hash=schema_hash,
            parent_version_id=parent_version_id,
            transformation_history=transformation_history or ["initial_ingestion"],
        )
