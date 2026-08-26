"""Multi-format file ingestion engine supporting CSV, XLSX, JSON, and Parquet."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import uuid4
import pandas as pd


class IngestionFileRecord:
    def __init__(self, dataset_id: str, path: Path, df: pd.DataFrame, file_type: str, file_hash: str):
        self.dataset_id = dataset_id
        self.path = path
        self.filename = path.name
        self.df = df
        self.file_type = file_type
        self.file_hash = file_hash
        self.file_size_bytes = path.stat().st_size if path.exists() else 0


class MultiFileIngestionEngine:
    """Ingests heterogeneous tabular datasets without modifying source files."""

    @staticmethod
    def compute_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def ingest_file(cls, file_path: str) -> IngestionFileRecord:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        ext = path.suffix.lower()
        file_hash = cls.compute_sha256(path)
        dataset_id = f"ds_{uuid4().hex[:6]}"

        if ext == ".csv":
            df = pd.read_csv(path)
            ftype = "csv"
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
            ftype = "excel"
        elif ext == ".json":
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
            df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame([data])
            ftype = "json"
        elif ext == ".parquet":
            df = pd.read_parquet(path)
            ftype = "parquet"
        else:
            # Fallback to CSV text reader
            df = pd.read_csv(path, sep=None, engine="python")
            ftype = "text_delimited"

        return IngestionFileRecord(
            dataset_id=dataset_id,
            path=path,
            df=df,
            file_type=ftype,
            file_hash=file_hash,
        )

    @classmethod
    def ingest_files(cls, file_paths: List[str]) -> List[IngestionFileRecord]:
        return [cls.ingest_file(p) for p in file_paths]
